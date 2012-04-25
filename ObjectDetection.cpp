#include "ObjectDetection.h"

// only splits on spaces
void SplitStr(std::string str, char sp, std::vector<std::string>& splits)
{
	std::string s;
	std::stringstream ss;

	ss<<str;
	while(ss >> s) {
		splits.push_back(s);
	}
	
}

SVMModule::SVMModule()
{
	m_threshold = 0.5;
	m_pTrainModel = NULL;
}

int SVMModule::LoadSVMModelFile(std::string modelFile)
{
        MODEL* model = read_model((char*)modelFile.c_str());
        if(model == NULL)
               return -1;
        add_weight_vector_to_linear_model(model);
	m_pTrainModel = model;
        return 0;
}

int SVMModule::GetSupportVector(std::vector<float>& supVector)
{
	if(m_pTrainModel == NULL)
		return -1;
	MODEL* model = m_pTrainModel;
	supVector.resize(model->totwords);

        for(int i = 1; i <= model->totwords; i++) {
                supVector[i-1] = model->lin_weights[i];
        }
        //supVector[0] = model->b;
        return 0;
}

double SVMModule::classify(std::vector<float>& instance)
{
	if(m_pTrainModel == NULL)
		return -1;

	WORD *words;
	int vecLen = instance.size();
	words = (WORD*)my_malloc(sizeof(WORD)*(vecLen + 1));

	for(int i = 0; i < vecLen; i++) {
		words[i].wnum = (i+1);
		words[i].weight = instance[i];
	}
	words[vecLen].wnum = 0;

	// make svector
	SVECTOR* svec = create_svector(words, "", 1.0); 
	DOC* doc = create_example(-1, 0, 0, 0.0, svec);

	double prob = classify_example(m_pTrainModel, doc);

	free_example(doc, 1);
	
	return prob;
}

ObjectDetection::ObjectDetection()
{
	m_threshold = 0.5;
	m_winHt = 64;
	m_winWid = 64;
}

void ObjectDetection::DumpToSVMFormat(std::vector<float>& rhog, bool classType)
{
        m_svmfile<<classType;
        for(int i = 0;i < rhog.size(); i++) {
                m_svmfile<<" " << (i+1) << ":" << rhog[i];
        }
        m_svmfile<<"\n";
}

int ObjectDetection::GetHogDescriptor(cv::Mat& img, std::vector<float> &hogDesc)
{
        int ht = img.cols;
        int width = img.rows;

	cv::HOGDescriptor hog(cv::Size(ht, width), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9, 1, -1,
                                                                cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);

        hog.compute(img, hogDesc, cv::Size(16, 16), cv::Size(0,0) );

        return 0;
}

int ObjectDetection::GetHSVHistogram(cv::Mat& img, std::vector<float>& histVec)
{
        cv::Mat hsv;
    	cv::cvtColor(img, hsv, CV_BGR2HSV);

        int hbins = 30, sbins = 32;
	int histSize[] = {hbins, sbins};
	// hue varies from 0 to 179, see cvtColor
    	float hranges[] = { 0, 180 };
    	// saturation varies from 0 (black-gray-white) to
    	// 255 (pure spectrum color)
    	float sranges[] = { 0, 256 };
    	const float* ranges[] = { hranges, sranges };
    	cv::MatND hist;

    	// we compute the histogram from the 0-th and 1-st channels
    	int channels[] = {0, 1};

        cv::calcHist( &hsv, 1, channels, cv::Mat(), // do not use mask
				        hist, 2, histSize, ranges,
				        true, // the histogram is uniform
					false );
        cv::normalize( hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat() );
	for(int i = 0; i < hist.rows; i++)
		for(int j = 0;j < hist.cols; j++)
			histVec.push_back(hist.at<float>(i,j));

	return 0;
}

int ObjectDetection::train(cv::Mat& img, bool classType)
{
        if(img.data == NULL)
                return -1;

        std::vector<float> hogDesc(9*4*7*15, 0);
        GetHogDescriptor(img, hogDesc);
	GetHSVHistogram(img, hogDesc);
        DumpToSVMFormat(hogDesc, classType);

        return 0;
}

int ObjectDetection::trainMultiScale(std::string imgName, bool classType)
{
	int stride = 32;
        // get the bounding box too if there
	// format of line - imageName #objs=1 x y x_width y_width
	// split
	std::vector<std::string> strs;
//	boost::split(strs, imgName, boost::is_any_of(" ") );
	SplitStr(imgName, ' ', strs);
	
	if( strs.empty() == true ) return -1;
	
	std::string imgStr = strs[0];

	cv::Mat img = cv::imread(imgStr.c_str());

        if(img.data == NULL)
                return -1;

        int ht = img.rows;
        int wid = img.cols;

        if( 4*ht < 3*m_winHt || 4*wid < 3*m_winWid ) {
		return 0;
        }

        // bounding box
        int x_left = 0, y_left = 0, x_wid = wid, y_ht = ht;
        if(strs.size() >= 6) {
                x_left = atoi(strs[2].c_str());
                y_left = atoi(strs[3].c_str());
                x_wid = atoi(strs[4].c_str());
                y_ht = atoi(strs[5].c_str());
        }

        if( 4*y_ht < 3*m_winHt || 4*x_wid < 3*m_winWid ) {
		return 0;
        }

        cv::Rect roi(x_left, y_left, x_wid, y_ht);
        //cv::Mat boundedImg = cvCreateImage(cvGetSize(&img), img->depth, img->nChannels);
        cv::Mat boundedImg = cv::Mat(img, roi);

	if(classType == 0) {
		// multiscale training on negative dataset 
		for(int x = 0; x < wid; x += stride) {
			for(int y = 0; y < ht; y += stride) {
				if( (x + m_winWid) > wid || (y + m_winHt) > ht)
					continue;
				// detect
				cv::Mat blockImg = boundedImg(cv::Range(y,y+m_winHt), cv::Range(x, x+m_winWid));
				this->train(blockImg, classType);
			}
		}
	} else {
		// image resize
		cv::Size sz(m_winWid, m_winHt);
		cv::resize(boundedImg, boundedImg, sz, 0, 0, cv::INTER_LINEAR); 

		ht = m_winHt;
		wid = m_winWid;

		cv::imshow("object detector", boundedImg);
	        int c = cv::waitKey(0) & 255;
		this->train(boundedImg, classType);
	}
        return 0;
}

int ObjectDetection::testClassify(cv::Mat& img, double& prob)
{
//        cv::imshow("object detector", img);
//        int c = cv::waitKey(0) & 255;

        if(img.data == NULL)
                return -1;

	if(img.rows < m_winHt || img.cols < m_winWid) {
		cout << "image in not 64x64 size" <<endl;
		return -1;
	}

        std::vector<cv::Rect> found, found_filtered;
        long sz = m_phog->getDescriptorSize();

	// get hog desc for image
	std::vector<float> hogDesc;
	this->GetHogDescriptor(img, hogDesc);
	this->GetHSVHistogram(img, hogDesc);

	// classify
	prob = m_pSvm->classify(hogDesc);
	cout << "Probability: " << prob << endl;
	int classification = (prob > m_threshold ? 1 : 0);
	return classification;
}

cv::Rect ObjectDetection::testClassifyMultiScale(cv::Mat& img, int stride, double& prob)
{
//	cv::imshow("object detector", img);
//        int c = cv::waitKey(0) & 255;

	prob = 0.0;
	int ht = img.rows;
	int wid = img.cols;

	// scale to 64 if one of the dim is less than 64
	if( ht < m_winHt || wid < m_winWid ) {
		int minzD = m_winWid;
                int minz = wid;
                if(ht*m_winWid < m_winHt*wid) { minz = ht; minzD = m_winHt; }

                double sc = ((minzD*1.0) / (double)minz);

                if(sc > 1.5) return cv::Rect(0,0,0,0);

                cv::Size sz(0,0);
                if(ht == minz) {
                        sz.height = m_winHt;
                        sz.width = (sc * img.cols);
                        cv::resize(img, img, sz, sc, 0, cv::INTER_LINEAR);
                } else {
                        sz.width = m_winWid;
                        sz.height = (sc * img.rows);
                        cv::resize(img, img, sz, 0, sc, cv::INTER_LINEAR);
                }
                ht = img.rows;
                wid = img.cols;
	}

	// multiscale detection - it calculates the max prob at diff scales
	double max_prob = 0.0;
	cv::Mat max_frame;
	cv::Rect max_frame_rect(0, 0, 0, 0);

	for(double scale = 1.0; scale <= 5.0; scale *= 1.2) {
		cv::Mat simg;
		cv::resize(img, simg, cv::Size(0, 0), (1.0/scale), (1.0/scale), cv::INTER_LINEAR);
		if( simg.rows < m_winHt || simg.cols < m_winWid)
			continue;
		int sht = simg.rows;
		int swid = simg.cols;
		for(int x = 0; x < swid; x += stride) {
			for(int y = 0; y < sht; y += stride) {
				if( (x + m_winWid) > swid || (y + m_winHt) > sht)
					continue;
				// detect
				double p = 0.0;
				cv::Mat blockImg = simg(cv::Range(y,y+m_winHt), cv::Range(x, x+m_winWid));
				this->testClassify(blockImg, p);
				if( p > max_prob) {
					max_prob = p;
					blockImg.copyTo(max_frame);
					max_frame_rect.x = (x * scale);
					max_frame_rect.y = (y * scale);
					max_frame_rect.width = (m_winWid * scale);
					max_frame_rect.height = (m_winHt * scale);
				}
			}
		}
	}
	prob = max_prob;
	cout << "Max probability: " << max_prob << endl;

	if(max_frame.data != NULL) {	
//		cv::imshow("object detector", max_frame);
//        	cv::waitKey(0) & 255;
	}
	
	return max_frame_rect;
}

int ObjectDetection::test(std::string imgName)
{
	std::vector<std::string> strs;
//        boost::split(strs, imgName, boost::is_any_of(" ") );
	SplitStr(imgName, ' ', strs);

        if( strs.empty() == true ) return -1;

        std::string imgStr = strs[0];

        cv::Mat img = cv::imread(imgStr.c_str());

        if(img.data == NULL)
                return -1;

        std::vector<cv::Rect> found, found_filtered;
        long sz = m_phog->getDescriptorSize();
        // run the detector with default parameters. to get a higher hit-rate
        // (and more false alarms, respectively), decrease the hitThreshold and
        // groupThreshold (set groupThreshold to 0 to turn off the grouping completely).
        //m_phog->detectMultiScale(img, found, 0, Size(16,16), Size(0, 0), 1.05, 0);
#if 0
	m_phog->detectMultiScale(img, found, 0.47, cv::Size(8,8), cv::Size(32,32), 1.05, 2);
        cv::namedWindow("object detector", 1);

        size_t i, j;
        for( i = 0; i < found.size(); i++ )
        {
                cv::Rect r = found[i];
                for( j = 0; j < found.size(); j++ )
                        if( j != i && (r & found[j]) == r)
                                break;
                if( j == found.size() )
                        found_filtered.push_back(r);
        }
        for( i = 0; i < found_filtered.size(); i++ )
        {
                cv::Rect r = found_filtered[i];
                // the HOG detector returns slightly larger rectangles than the real objects.
                // so we slightly shrink the rectangles to get a nicer output.
                r.x += cvRound(r.width*0.1);
                r.width = cvRound(r.width*0.8);
                r.y += cvRound(r.height*0.07);
                r.height = cvRound(r.height*0.8);
                cv::rectangle(img, r.tl(), r.br(), cv::Scalar(0,255,0), 3);
        }

        cv::imshow("object detector", img);
	int c = cv::waitKey(0) & 255;
#endif
#if 1
	double prob = 0.0;
//	int classification = testClassify(img, prob);
	testClassifyMultiScale(img, 16, prob);
	cout<<"Classified: " << prob <<endl;
#endif	
	return 0;
}

int ObjectDetection::InitTest(std::string modelName)
{
        std::vector<float> supVector;

	std::string testingFile = "testSvmOpencv_";
        testingFile += m_objectName;
        testingFile += ".txt";
	m_svmfile.open (testingFile.c_str());//"testSvmOpencv_banana.txt");

	SVMModule* psvm = new SVMModule();
	m_pSvm = psvm;
	psvm->LoadSVMModelFile(modelName);
	psvm->GetSupportVector(supVector);

        /*HOGDescriptor hog(Size(64, 64), Size(16,16), Size(8,8), Size(8,8), 9, 1, -1,
                                                                HOGDescriptor::L2Hys, 0.2, true, HOGDescriptor::DEFAULT_NLEVELS);*/
        m_phog = new cv::HOGDescriptor(cv::Size(m_winWid, m_winHt), cv::Size(16,16), cv::Size(8,8), cv::Size(8,8), 9, 1, -1,
                                                                cv::HOGDescriptor::L2Hys, 0.2, true, cv::HOGDescriptor::DEFAULT_NLEVELS);
//	m_phog->setSVMDetector(supVector);


        return 0;
}

int ObjectDetection::InitTrain()
{
	std::string trainingFile = "trainSvmOpencv_";
	trainingFile += m_objectName;
	trainingFile += ".txt";
	m_svmfile.open(trainingFile.c_str());//"trainSvmOpencv_pear.txt");
	return 0;
}

int ObjectDetection::Close()
{
	m_svmfile.close();
}

#ifndef NOSTANDALONE_DETECTION
int main( int argc, const char** argv )
{
    	cv::Mat img;
	std::string posFileName = "";
	std::string negFileName = "";

	ObjectDetection* objDetect = new ObjectDetection();
	objDetect->SetObjectName(argv[4]);

	if(argc == 1) {
		printf("No arguments received\n");
		return 0;
	}
	
	// window size
	int width = atoi(argv[5]);
	int height = atoi(argv[6]);

	objDetect->SetWindowSize(height, width);

	bool bTrain = true;
	if(strcmp("test", argv[1]) == 0)
		bTrain = false;

	if(bTrain == true) {
		posFileName = argv[2];
		negFileName = argv[3];
		objDetect->InitTrain();
	}
	else {
		posFileName = argv[2];

		//MODEL* trainModel = objDetect->LoadSVMModelFile(argv[3]);
		objDetect->InitTest(argv[3]);
	}


	// get the file that contains the file list
	std::ifstream fileList (posFileName.c_str());
	if (fileList.is_open())
	{
		std::string line;
		while ( fileList.good() )
		{
			std::getline (fileList,line);
			cout << line << endl;
			
			if(bTrain == true)
				objDetect->trainMultiScale(line, 1);
			else
				objDetect->test(line);
		}
		fileList.close();
	}

	// get the file that contains the file list
	if(negFileName.empty() == false) {
		std::ifstream negList (negFileName.c_str());
		if (negList.is_open())
		{
			std::string line;
			while ( negList.good() )
			{
				std::getline (negList,line);
				cout << line << endl;

				objDetect->trainMultiScale(line, 0);
			}
			negList.close();
		}
	}

	objDetect->Close();
	return 0;
}
#endif


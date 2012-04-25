//#include <opencv/cv.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <pcl/tracking/tracking.h>
#include <pcl/tracking/particle_filter.h>
#include <pcl/tracking/kld_adaptive_particle_filter_omp.h>
#include <pcl/tracking/particle_filter_omp.h>

#include <pcl/tracking/coherence.h>
#include <pcl/tracking/distance_coherence.h>
#include <pcl/tracking/hsv_color_coherence.h>
#include <pcl/tracking/normal_coherence.h>

#include <pcl/tracking/approx_nearest_pair_point_cloud_coherence.h>
#include <pcl/tracking/nearest_pair_point_cloud_coherence.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/console/parse.h>
#include <pcl/common/time.h>
#include <pcl/common/centroid.h>

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/passthrough.h>
#include <pcl/filters/project_inliers.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/approximate_voxel_grid.h>
#include <pcl/filters/extract_indices.h>

#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/integral_image_normal.h>

#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>

#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_polygonal_prism_data.h>
#include <pcl/segmentation/extract_clusters.h>

#include <pcl/surface/convex_hull.h>

#include <pcl/search/pcl_search.h>
#include <pcl/common/transforms.h>

#include <boost/format.hpp>

#include <algorithm>

#include <pcl/console/parse.h>
#include <pcl/console/print.h>
#include <pcl/io/pcd_io.h>
#include <iostream>
#include <flann/flann.h>
#include <flann/io/hdf5.h>
#include <boost/filesystem.hpp>

#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/features/vfh.h>

#include <pcl/visualization/image_viewer.h>
#include <pcl/range_image/range_image.h>
#include <pcl/visualization/common/float_image_utils.h>
#include "ObjectDetection.h"
#include "KalmanFilter.h"

#ifdef True
#undef True
#endif

#ifdef False
#undef False
#endif

#include <fstream>
#include <string>

//using namespace cv;
using namespace std;

#define FH_FEATURE 1
#define OPENCV 0
#define CAMERA_INPUT 0

#define FPS_CALC_BEGIN                          \
    static double duration = 0;                 \
    double start_time = pcl::getTime ();        \

#define FPS_CALC_END(_WHAT_)                    \
  {                                             \
    double end_time = pcl::getTime ();          \
    static unsigned count = 0;                  \
    if (++count == 10)                          \
    {                                           \
      std::cout << "Average framerate("<< _WHAT_ << "): " << double(count)/double(duration) << " Hz" <<  std::endl; \
      count = 0;                                                        \
      duration = 0.0;                                                   \
    }                                           \
    else                                        \
    {                                           \
      duration += end_time - start_time;        \
    }                                           \
  }


using namespace pcl::tracking;
typedef std::pair<std::string, std::vector<float> > vfh_model;
 
struct CirclePoints {
	pcl::PointXYZ center_pt;
	float rad;
	pcl::PointXYZRGB seg_min, seg_max;
};

struct Clusters {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr pCluster;
	int type;
	double score;
};

struct BoundingBox {
	double x,y,z;
	double height, width;
	double prob;

	BoundingBox(){x=0.0,y=0.0,z=0.0,height=0.0,width=0.0,prob=0.0;}
};

template <typename PointType>
class TestSegmentTracking
{
public:
  //typedef pcl::PointXYZRGBNormal RefPointType;
  typedef pcl::PointXYZRGB RefPointType;
  //typedef pcl::PointXYZ RefPointType;
  typedef ParticleXYZRPY ParticleT;
  
  typedef pcl::PointCloud<PointType> Cloud;
  typedef pcl::PointCloud<RefPointType> RefCloud;
  typedef typename RefCloud::Ptr RefCloudPtr;
  typedef typename RefCloud::ConstPtr RefCloudConstPtr;
  typedef typename Cloud::Ptr CloudPtr;
  typedef typename Cloud::ConstPtr CloudConstPtr;
  //typedef KLDAdaptiveParticleFilterTracker<RefPointType, ParticleT> ParticleFilter;
  //typedef KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> ParticleFilter;
  //typedef ParticleFilterOMPTracker<RefPointType, ParticleT> ParticleFilter;
  typedef ParticleFilterTracker<RefPointType, ParticleT> ParticleFilter;
  typedef typename ParticleFilter::CoherencePtr CoherencePtr;
  typedef typename pcl::search::KdTree<PointType> KdTree;
  typedef typename KdTree::Ptr KdTreePtr;
  TestSegmentTracking (const std::string& device_id, int thread_nr, double downsampling_grid_size,
                         bool use_convex_hull,
                         bool visualize_non_downsample, bool visualize_particles,
                         bool use_fixed, std::string objName)
  : viewer_ ("PCL Test Tracking Viewer")
  , device_id_ (device_id)
  , new_cloud_ (false)
  , ne_ (thread_nr)
  , counter_ (0)
  , use_convex_hull_ (use_convex_hull)
  , visualize_non_downsample_ (visualize_non_downsample)
  , visualize_particles_ (visualize_particles)
  , downsampling_grid_size_ (downsampling_grid_size)
	, m_objectName(objName)
  {
    KdTreePtr tree (new KdTree (false));
    ne_.setSearchMethod (tree);
    ne_.setRadiusSearch (0.03);
    
    std::vector<double> default_step_covariance = std::vector<double> (6, 0.015 * 0.015);
    default_step_covariance[3] *= 40.0;
    default_step_covariance[4] *= 40.0;
    default_step_covariance[5] *= 40.0;
    
    std::vector<double> initial_noise_covariance = std::vector<double> (6, 0.00001);
    std::vector<double> default_initial_mean = std::vector<double> (6, 0.0);
    if (use_fixed)
    {
      boost::shared_ptr<ParticleFilterOMPTracker<RefPointType, ParticleT> > tracker
        (new ParticleFilterOMPTracker<RefPointType, ParticleT> (thread_nr));
      tracker_ = tracker;
    }
    else
    {
      boost::shared_ptr<KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> > tracker
        (new KLDAdaptiveParticleFilterOMPTracker<RefPointType, ParticleT> (thread_nr));
      tracker->setMaximumParticleNum (500);
      tracker->setDelta (0.99);
      tracker->setEpsilon (0.2);
      ParticleT bin_size;
      bin_size.x = 0.1;
      bin_size.y = 0.1;
      bin_size.z = 0.1;
      bin_size.roll = 0.1;
      bin_size.pitch = 0.1;
      bin_size.yaw = 0.1;
      tracker->setBinSize (bin_size);
      tracker_ = tracker;
    }
    
    tracker_->setTrans (Eigen::Affine3f::Identity ());
    tracker_->setStepNoiseCovariance (default_step_covariance);
    tracker_->setInitialNoiseCovariance (initial_noise_covariance);
    tracker_->setInitialNoiseMean (default_initial_mean);
    tracker_->setIterationNum (1);
    
    tracker_->setParticleNum (400);
    tracker_->setResampleLikelihoodThr(0.00);
    tracker_->setUseNormal (false);
    // setup coherences
    ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr coherence = ApproxNearestPairPointCloudCoherence<RefPointType>::Ptr
      (new ApproxNearestPairPointCloudCoherence<RefPointType> ());
    // NearestPairPointCloudCoherence<RefPointType>::Ptr coherence = NearestPairPointCloudCoherence<RefPointType>::Ptr
    //   (new NearestPairPointCloudCoherence<RefPointType> ());
    
    boost::shared_ptr<DistanceCoherence<RefPointType> > distance_coherence
      = boost::shared_ptr<DistanceCoherence<RefPointType> > (new DistanceCoherence<RefPointType> ());
    coherence->addPointCoherence (distance_coherence);
    
    boost::shared_ptr<HSVColorCoherence<RefPointType> > color_coherence
      = boost::shared_ptr<HSVColorCoherence<RefPointType> > (new HSVColorCoherence<RefPointType> ());
    color_coherence->setWeight (0.1);
    coherence->addPointCoherence (color_coherence);
    
    //boost::shared_ptr<pcl::search::KdTree<RefPointType> > search (new pcl::search::KdTree<RefPointType> (false));
    boost::shared_ptr<pcl::search::Octree<RefPointType> > search (new pcl::search::Octree<RefPointType> (0.01));
    //boost::shared_ptr<pcl::search::OrganizedNeighbor<RefPointType> > search (new pcl::search::OrganizedNeighbor<RefPointType>);
    coherence->setSearchMethod (search);
    coherence->setMaximumDistance (0.01);
    tracker_->setCloudCoherence (coherence);

		m_pObjectDetector = new ObjectDetection();
		m_pObjectDetector->SetObjectName(m_objectName);

 		std::string modelName = "trainSvmOpencv_";
  	modelName += m_objectName;
    modelName += ".txt.model";
    m_pObjectDetector->InitTest(modelName.c_str());//"trainSvmOpencv_banana.txt.model");
    m_pObjectDetector->SetWindowSize(64, 96);

    Init3DObjDetector();
    m_kalman.Init();

#if OPENCV
    OpenCVInit();
#endif
  }

  bool
  drawParticles (pcl::visualization::PCLVisualizer& viz)
  {
    ParticleFilter::PointCloudStatePtr particles = tracker_->getParticles ();
    if (particles)
    {
      if (visualize_particles_)
      {
        pcl::PointCloud<pcl::PointXYZ>::Ptr particle_cloud (new pcl::PointCloud<pcl::PointXYZ> ());
        for (size_t i = 0; i < particles->points.size (); i++)
        {
          pcl::PointXYZ point;
          
          point.x = particles->points[i].x;
          point.y = particles->points[i].y;
          point.z = particles->points[i].z;
          particle_cloud->points.push_back (point);
        }
        
        {
          pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> blue_color (particle_cloud, 250, 99, 71);
          if (!viz.updatePointCloud (particle_cloud, blue_color, "particle cloud"))
            viz.addPointCloud (particle_cloud, blue_color, "particle cloud");
        }
      }
      return true;
    }
    else
    {
      PCL_WARN ("no particles\n");
      return false;
    }
  }
  
  void
  drawResult (pcl::visualization::PCLVisualizer& viz)
  {
    ParticleXYZRPY result = tracker_->getResult ();
    Eigen::Affine3f transformation = tracker_->toEigenMatrix (result);
    // move a little bit for better visualization
    transformation.translation () += Eigen::Vector3f (0.0, 0.0, -0.005);
    RefCloudPtr result_cloud (new RefCloud ());

    if (!visualize_non_downsample_)
      pcl::transformPointCloud<RefPointType> (*(tracker_->getReferenceCloud ()), *result_cloud, transformation);
    else
      pcl::transformPointCloud<RefPointType> (*reference_, *result_cloud, transformation);

    {
      pcl::visualization::PointCloudColorHandlerCustom<RefPointType> red_color (result_cloud, 0, 0, 255);
      if (!viz.updatePointCloud (result_cloud, red_color, "resultcloud"))
        viz.addPointCloud (result_cloud, red_color, "resultcloud");
    }
    
  }
  
  void
  viz_cb (pcl::visualization::PCLVisualizer& viz)
  {
    boost::mutex::scoped_lock lock (mtx_);
    
    if (!cloud_pass_)
    {
      boost::this_thread::sleep (boost::posix_time::seconds (1));
      return;
    }
    
    if (new_cloud_ && cloud_pass_downsampled_)
    {
//    viz.removeAllShapes();
      CloudPtr cloud_pass;
      if (!visualize_non_downsample_)
        cloud_pass = cloud_pass_downsampled_;
      else
        cloud_pass = cloud_pass_;

#if 0
//	cloud_pass = curr_cloud_;
	drawCube (viz, curr_cloud_);
#endif


#if OPENCV
	vector<CirclePoints> detObjs = detectedObjsXYZ_;
	for(int i = 0; i < detObjs.size(); i++) {
		pcl::ModelCoefficients circle_coeff;
		circle_coeff.values.resize(3);
		circle_coeff.values[0] = detObjs[i].center_pt.x;
		circle_coeff.values[0] = detObjs[i].center_pt.y;
		circle_coeff.values[0] = detObjs[i].center_pt.z;
		circle_coeff.values[0] = detObjs[i].rad;
		std::string sid = "circle";
		sid += i;
		cout << "Adding circle" << detObjs[i].center_pt.x << "," << detObjs[i].center_pt.y << "," << detObjs[i].center_pt.z << ", "  << detObjs[i].rad << endl;		
		//viz.addCircle(circle_coeff, sid.c_str());
		viz.addSphere(detObjs[i].center_pt, detObjs[i].rad, sid.c_str());
		drawCube(viz, detObjs[i].seg_min, detObjs[i].seg_max);
	}
	detectedObjsXYZ_.clear();
#endif
#if FH_FEATURE
	// show labelled clusters
	Clusters c;
	c.score = 0.0;
	vector<Clusters> objLabels = segmentedClusters_;
	if(objLabels.size() > 0)
		viz.removeAllShapes();

	for(int i = 0; i < objLabels.size(); i++) {
		cout<<"Drawing Cube for Object: " << objLabels[i].score<<endl;
		std::stringstream ss;
		ss<<objLabels[i].score;
		objLabels[i].score;
		drawCube(viz, (objLabels[i].pCluster), ss.str());
		if(objLabels[i].score > c.score) 
			c = objLabels[i];
	}
	//std::stringstream ss;
	//ss<<c.score;

	/*if(c.score >= 0.5) {
		drawCube(viz, c.pCluster, ss.str());
		cout<<"Drawing Cube for Object: " << c.score<<endl;
	}*/

	// show kalman predicted Sphere
	printf("Drawing the Predicted Kalman Values\n");
	viz.addSphere(m_predictedPoint, 0.03, "Kalman");

        segmentedClusters_.clear();
#endif
      if (!viz.updatePointCloud (cloud_pass, "cloudpass"))
        {
          viz.addPointCloud (cloud_pass, "cloudpass");
          viz.resetCameraViewpoint ("cloudpass");
        }
    }

#if 1
    if (new_cloud_ && reference_)
    {
 //     bool ret = drawParticles (viz);
   //   if (ret)
      {
        drawResult (viz);
      }
    }
#endif 
    new_cloud_ = false;
  }

  void filterPassThrough (const CloudConstPtr &cloud, Cloud &result)
  {
    FPS_CALC_BEGIN;
    pcl::PassThrough<PointType> pass;
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (0.0, 2.0);
    //pass.setFilterLimits (0.0, 1.5);
    //pass.setFilterLimits (0.0, 0.6);
    pass.setKeepOrganized (false);
    pass.setInputCloud (cloud);
    pass.filter (result);
    FPS_CALC_END("filterPassThrough");
  }

 
  void GetBoundedCloud (const CloudConstPtr &cloud, double x, double y, double z, double width, double height, double depth, Cloud &result)
  {
    pcl::PassThrough<PointType> pass;
    pass.setFilterFieldName ("z");
    pass.setFilterLimits (z, z+depth);
    pass.setFilterFieldName ("x");
    pass.setFilterLimits (x, x+width);
    pass.setFilterFieldName ("y");
    pass.setFilterLimits (y, y+height);
    pass.setKeepOrganized (false);
    pass.setInputCloud (cloud);
    pass.filter (result);
  }

  void euclideanSegment (const CloudConstPtr &cloud,
                         std::vector<pcl::PointIndices> &cluster_indices)
  {
    FPS_CALC_BEGIN;
    pcl::EuclideanClusterExtraction<PointType> ec;
    KdTreePtr tree (new KdTree ());
    
    ec.setClusterTolerance (0.05); // 2cm
    ec.setMinClusterSize (50);
    ec.setMaxClusterSize (25000);
    //ec.setMaxClusterSize (400);
    ec.setSearchMethod (tree);
    ec.setInputCloud (cloud);
    ec.extract (cluster_indices);
    FPS_CALC_END("euclideanSegmentation");
  }
  
  void gridSample (const CloudConstPtr &cloud, Cloud &result, double leaf_size = 0.01)
  {
    FPS_CALC_BEGIN;
    double start = pcl::getTime ();
    pcl::VoxelGrid<PointType> grid;
    //pcl::ApproximateVoxelGrid<PointType> grid;
    grid.setLeafSize (leaf_size, leaf_size, leaf_size);
    grid.setInputCloud (cloud);
    grid.filter (result);
    //result = *cloud;
    double end = pcl::getTime ();
    downsampling_time_ = end - start;
    FPS_CALC_END("gridSample");
  }
  
  void gridSampleApprox (const CloudConstPtr &cloud, Cloud &result, double leaf_size = 0.01)
  {
    FPS_CALC_BEGIN;
    double start = pcl::getTime ();
    //pcl::VoxelGrid<PointType> grid;
    pcl::ApproximateVoxelGrid<PointType> grid;
    grid.setLeafSize (leaf_size, leaf_size, leaf_size);
    grid.setInputCloud (cloud);
    grid.filter (result);
    //result = *cloud;
    double end = pcl::getTime ();
    downsampling_time_ = end - start;
    FPS_CALC_END("gridSample");
  }
  
  void planeSegmentation (const CloudConstPtr &cloud,
                          pcl::ModelCoefficients &coefficients,
                          pcl::PointIndices &inliers)
  {
    FPS_CALC_BEGIN;
    pcl::SACSegmentation<PointType> seg;
    seg.setOptimizeCoefficients (true);
    seg.setModelType (pcl::SACMODEL_PLANE);
    seg.setMethodType (pcl::SAC_RANSAC);
    seg.setMaxIterations (1000);
    seg.setDistanceThreshold (0.03);
    seg.setInputCloud (cloud);
    seg.segment (inliers, coefficients);
    FPS_CALC_END("planeSegmentation");
  }

  void planeProjection (const CloudConstPtr &cloud,
                        Cloud &result,
                        const pcl::ModelCoefficients::ConstPtr &coefficients)
  {
    FPS_CALC_BEGIN;
    pcl::ProjectInliers<PointType> proj;
    proj.setModelType (pcl::SACMODEL_PLANE);
    proj.setInputCloud (cloud);
    proj.setModelCoefficients (coefficients);
    proj.filter (result);
    FPS_CALC_END("planeProjection");
  }

  void convexHull (const CloudConstPtr &cloud,
                   Cloud &result,
                   std::vector<pcl::Vertices> &hull_vertices)
  {
    FPS_CALC_BEGIN;
    pcl::ConvexHull<PointType> chull;
    chull.setInputCloud (cloud);
    chull.reconstruct (*cloud_hull_, hull_vertices);
    FPS_CALC_END("convexHull");
  }

  void normalEstimation (const CloudConstPtr &cloud,
                         pcl::PointCloud<pcl::Normal> &result)
  {
    FPS_CALC_BEGIN;
    ne_.setInputCloud (cloud);
    ne_.compute (result);
    FPS_CALC_END("normalEstimation");
  }
  
  void tracking (const RefCloudConstPtr &cloud)
  {
    double start = pcl::getTime ();
    FPS_CALC_BEGIN;
    tracker_->setInputCloud (cloud);
    tracker_->compute ();
    double end = pcl::getTime ();
    FPS_CALC_END("tracking");
    tracking_time_ = end - start;
  }

  void addNormalToCloud (const CloudConstPtr &cloud,
                         const pcl::PointCloud<pcl::Normal>::ConstPtr &normals,
                         RefCloud &result)
  {
    result.width = cloud->width;
    result.height = cloud->height;
    result.is_dense = cloud->is_dense;
    for (size_t i = 0; i < cloud->points.size (); i++)
    {
      RefPointType point;
      point.x = cloud->points[i].x;
      point.y = cloud->points[i].y;
      point.z = cloud->points[i].z;
      point.rgb = cloud->points[i].rgb;
      // point.normal[0] = normals->points[i].normal[0];
      // point.normal[1] = normals->points[i].normal[1];
      // point.normal[2] = normals->points[i].normal[2];
      result.points.push_back (point);
    }
  }

  void extractNonPlanePoints (const CloudConstPtr &cloud,
                              const CloudConstPtr &cloud_hull,
                              Cloud &result)
  {
    pcl::ExtractPolygonalPrismData<PointType> polygon_extract;
    pcl::PointIndices::Ptr inliers_polygon (new pcl::PointIndices ());
    polygon_extract.setHeightLimits (0.01, 10.0);
    polygon_extract.setInputPlanarHull (cloud_hull);
    polygon_extract.setInputCloud (cloud);
    polygon_extract.segment (*inliers_polygon);
    {
      pcl::ExtractIndices<PointType> extract_positive;
      extract_positive.setNegative (false);
      extract_positive.setInputCloud (cloud);
      extract_positive.setIndices (inliers_polygon);
      extract_positive.filter (result);
    }
  }

  void removeZeroPoints (const CloudConstPtr &cloud,
                         Cloud &result)
  {
    for (size_t i = 0; i < cloud->points.size (); i++)
    {
      PointType point = cloud->points[i];
      if (!(fabs(point.x) < 0.01 &&
            fabs(point.y) < 0.01 &&
            fabs(point.z) < 0.01) &&
          !pcl_isnan(point.x) &&
          !pcl_isnan(point.y) &&
          !pcl_isnan(point.z))
        result.points.push_back(point);
    }

    result.width = result.points.size ();
    result.height = 1;
    result.is_dense = true;
  }
  
  void extractSegmentCluster (const CloudConstPtr &cloud,
                              const std::vector<pcl::PointIndices> cluster_indices,
                              const int segment_index,
                              Cloud &result)
  {
    pcl::PointIndices segmented_indices = cluster_indices[segment_index];
    for (size_t i = 0; i < segmented_indices.indices.size (); i++)
    {
      PointType point = cloud->points[segmented_indices.indices[i]];
      result.points.push_back (point);
    }
    result.width = result.points.size ();
    result.height = 1;
    result.is_dense = true;
  }
  
  void
  cloud_cb (const CloudConstPtr &cloud)
  {
    boost::mutex::scoped_lock lock (mtx_);
    double start = pcl::getTime ();
    FPS_CALC_BEGIN;
    curr_cloud_.reset(new Cloud);
    cloud_pass_.reset (new Cloud);
    cloud_pass_downsampled_.reset (new Cloud);
    pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
    pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());

    filterPassThrough (cloud, *cloud_pass_);

		GetImage(cloud);

    curr_cloud_ = cloud_pass_;

    gridSample (cloud_pass_, *cloud_pass_downsampled_, 0.01);
    cloud_pass_downsampled_ = cloud_pass_;

#if 1
    if (counter_%2 == 0)
#endif
    {
      CloudPtr target_cloud;
      if (use_convex_hull_)
      {
        planeSegmentation (cloud_pass_downsampled_, *coefficients, *inliers);
        if (inliers->indices.size () > 3)
        {
          CloudPtr cloud_projected (new Cloud);
          cloud_hull_.reset (new Cloud);
          nonplane_cloud_.reset (new Cloud);
          
          planeProjection (cloud_pass_downsampled_, *cloud_projected, coefficients);
          convexHull (cloud_projected, *cloud_hull_, hull_vertices_);
          
          extractNonPlanePoints (cloud_pass_downsampled_, cloud_hull_, *nonplane_cloud_);
          target_cloud = nonplane_cloud_;

	  curr_cloud_ = nonplane_cloud_;
	  	
        }
        else
        {
          PCL_WARN ("cannot segment plane\n");
        }
      }
      else
      {
        PCL_WARN ("without plane segmentation\n");
        target_cloud = cloud_pass_downsampled_;
      }
      
      if (target_cloud != NULL)
      {
        PCL_INFO ("segmentation, please wait...\n");
        std::vector<pcl::PointIndices> cluster_indices;
        euclideanSegment (target_cloud, cluster_indices);
        if (cluster_indices.size () > 0)
        {
          // select the cluster to track
          CloudPtr temp_cloud (new Cloud);
          int segment_index = 0;
				
          LabelClusters(target_cloud, cluster_indices);
          
/*          segmented_cloud_.reset (new Cloud);
          extractSegmentCluster (target_cloud, cluster_indices, segment_index, *segmented_cloud_);
          RefCloudPtr ref_cloud (new RefCloud);
          ref_cloud = segmented_cloud_;
          RefCloudPtr nonzero_ref (new RefCloud);
          removeZeroPoints (ref_cloud, *nonzero_ref);
          
				  curr_cloud_ = nonzero_ref;
*/      
#if 0    
          RefCloudPtr transed_ref (new RefCloud);
          pcl::compute3DCentroid<RefPointType> (*nonzero_ref, c);
          Eigen::Affine3f trans = Eigen::Affine3f::Identity ();
          trans.translation () = Eigen::Vector3f (c[0], c[1], c[2]);
          //pcl::transformPointCloudWithNormals<RefPointType> (*ref_cloud, *transed_ref, trans.inverse());
          pcl::transformPointCloud<RefPointType> (*nonzero_ref, *transed_ref, trans.inverse());
          CloudPtr transed_ref_downsampled (new Cloud);
          gridSample (transed_ref, *transed_ref_downsampled, downsampling_grid_size_);
          tracker_->setReferenceCloud (transed_ref_downsampled);
          tracker_->setTrans (trans);
          reference_ = transed_ref;
          tracker_->setMinIndices (ref_cloud->points.size () / 2);
#endif
        }
        else
        {
          PCL_WARN ("euclidean segmentation failed\n");
        }
      }
    }
#if 1 
    else
    {
	m_predictedPoint = m_kalman.Correct(NULL, NULL, false);    	
      //normals_.reset (new pcl::PointCloud<pcl::Normal>);
      //normalEstimation (cloud_pass_downsampled_, *normals_);
      //RefCloudPtr tracking_cloud (new RefCloud ());
      //addNormalToCloud (cloud_pass_downsampled_, normals_, *tracking_cloud);
      //tracking_cloud = cloud_pass_downsampled_;
      
      //*cloud_pass_downsampled_ = *cloud_pass_;
      //cloud_pass_downsampled_ = cloud_pass_;
/*      gridSampleApprox (cloud_pass_, *cloud_pass_downsampled_, downsampling_grid_size_);
      tracking (cloud_pass_downsampled_);
*/    }
  
#endif
 
    new_cloud_ = true;
    double end = pcl::getTime ();
    computation_time_ = end - start;
    FPS_CALC_END("computation");
    counter_++;

#if 0
    viz_cb();
#endif
  }

	void LabelClusters(CloudPtr segmentedCloud, std::vector<pcl::PointIndices>& cluster_indices)
	{
		segmentedClusters_.clear();
		CloudPtr temp_cloud (new Cloud);
		Clusters maxCluster;
		maxCluster.score = 0;

		for (size_t i = 1; i < cluster_indices.size (); i++)
    		{
			Clusters c;
			c.score = 0;
			
			// Get the scores based on 3D features
    			temp_cloud.reset (new Cloud);
			extractSegmentCluster (segmentedCloud, cluster_indices, i, *temp_cloud);

			// get score on 3D
			//double score3d 
			BoundingBox box3d = LabelCluster3D(temp_cloud);

			// Get the scores based on 2D features
			//double score2d 
			BoundingBox box2d = LabelCluster2D(temp_cloud);

			double thresholdScale = 3.0;
			// if 3d bounding box is much wider then consider only 2d
/*			if(box3d.height * thresholdScale > box2d.height && box3d.width * thresholdScale > box2d.width) {
				printf("Pruning the cluster\n");
				CloudPtr bbCloud(new Cloud);
				GetBoundedCloud(cloud_pass_, box2d.x, box2d.y, box3d.z, box2d.width, box2d.height, box2d.width, *temp_cloud);
				box3d = LabelCluster3D(temp_cloud);
			}
*/
			double score3d = box3d.prob;
			double score2d = box2d.prob;

			cout << "Clusters scores - 3D: " << score3d << " 2D: " << score2d <<endl;
      			//if( objType.type == 1 ) { // banana
			c.pCluster = temp_cloud;
			c.score = score2d + score3d;
    			segmentedClusters_.push_back(c);

      			//}
			if(c.score > maxCluster.score) {
				maxCluster.pCluster = temp_cloud;
				maxCluster.score = c.score;
			}
    		}

		if(maxCluster.score > 0) {
			pcl::PointXYZRGB min_pt, max_pt;
    			pcl::getMinMax3D(maxCluster.pCluster.operator*(), min_pt, max_pt);

			pcl::PointXYZ m_pt;
			m_pt.x = min_pt.x, m_pt.y = min_pt.y, m_pt.z = min_pt.z;
			pcl::PointXYZ mx_pt;
			mx_pt.x = max_pt.x, mx_pt.y = max_pt.y, mx_pt.z = max_pt.z;
			// predicted kalman
			m_predictedPoint = m_kalman.Correct(&m_pt, &mx_pt, true);
		} else {
			m_predictedPoint = m_kalman.Correct(NULL, NULL, false);
		}
	}

  void run ()
  {
#if CAMERA_INPUT
		pcl::Grabber* interface = new pcl::OpenNIGrabber(device_id_);
		boost::function<void (const CloudConstPtr&)> f =
															      boost::bind (&TestSegmentTracking::cloud_cb, this, _1);
    interface->registerCallback (f);
		interface->start();
#else
    boost::thread pcdThread(&TestSegmentTracking::FileGrabber, this);
#endif

#if 1
    viewer_.runOnVisualizationThread (boost::bind(&TestSegmentTracking::viz_cb, this, _1), "viz_cb");
#endif    
      
    while (!viewer_.wasStopped ())
      boost::this_thread::sleep(boost::posix_time::seconds(1));

#if CAMERA_INPUT
		interface->stop();
#endif

  }

  void FileGrabber()
  {
	string fileName;
	ifstream file ("bananaPcd.txt"/*"pcdFiles.txt"*/);

	// initialize the object detection with the training model
//	std::string modelName = "trainSvmOpencv_";
//	modelName += m_objectName;
//	modelName += ".txt.model";
//	m_pObjectDetector->InitTest(modelName.c_str());//"trainSvmOpencv_banana.txt.model");

//	Init3DObjDetector();

	if (file.is_open()) {
		
		while (file.good()) {
			getline(file, fileName);
			cout << fileName << endl;
			
			// grab the pcd file
			viewPcd(fileName);
				
			// just for opening one file
			//break;


			sleep(0.5);
		}
		cout << "All pcd's Processed" << endl;
	}

  }

  int viewPcd(string pcdName) 
  {
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);
	if (pcl::io::loadPCDFile<pcl::PointXYZRGB> (pcdName.c_str(), *cloud) == -1) //* load the file
 	{
    		PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
    		return (-1);
  	}

	// segmentation here
	cloud_cb(cloud);

//	DetectObject(cloud);

/*	// visualize
	pcl::visualization::CloudViewer viewer("Test Viewer");
	viewer.showCloud(curr_cloud_);
	while (!viewer.wasStopped())
	{
		sleep(1);
	}
*/
	return 0;	
  }

  void drawCube(pcl::visualization::PCLVisualizer& viz, const CloudConstPtr &targetCloud)
  {
	if(targetCloud == NULL) return;

	viz.removeAllShapes();

	pcl::ModelCoefficients coeff_cube;
	PointType min_pt, max_pt;
	pcl::getMinMax3D(targetCloud.operator*(), min_pt, max_pt);

	coeff_cube.values.resize(10);
	coeff_cube.values[0] = min_pt.x + (max_pt.x - min_pt.x)/2;
	coeff_cube.values[1] = min_pt.y + (max_pt.y - min_pt.y)/2;
	coeff_cube.values[2] = min_pt.z + (max_pt.z - min_pt.z)/2;
	coeff_cube.values[3] = 0;
	coeff_cube.values[4] = 0;
	coeff_cube.values[5] = 0;
	coeff_cube.values[6] = 1;
	coeff_cube.values[7] = fabs(max_pt.x - min_pt.x);
	coeff_cube.values[8] = fabs(max_pt.x - min_pt.x);
	coeff_cube.values[9] = fabs(max_pt.x - min_pt.x);

	viz.addCube(coeff_cube, "cube");
  }

  void drawCube(pcl::visualization::PCLVisualizer& viz, pcl::PointCloud<pcl::PointXYZRGB>::Ptr &targetCloud, std::string name)
  {
        if(targetCloud == NULL) return;

//        viz.removeAllShapes();

        pcl::ModelCoefficients coeff_cube;
        pcl::PointXYZRGB min_pt, max_pt;
        pcl::getMinMax3D(targetCloud.operator*(), min_pt, max_pt);

        coeff_cube.values.resize(10);
        coeff_cube.values[0] = min_pt.x + (max_pt.x - min_pt.x)/2;
        coeff_cube.values[1] = min_pt.y + (max_pt.y - min_pt.y)/2;
        coeff_cube.values[2] = min_pt.z + (max_pt.z - min_pt.z)/2;
        coeff_cube.values[3] = 0;
        coeff_cube.values[4] = 0;
        coeff_cube.values[5] = 0;
        coeff_cube.values[6] = 1;
        coeff_cube.values[7] = fabs(max_pt.x - min_pt.x);
        coeff_cube.values[8] = fabs(max_pt.x - min_pt.x);
        coeff_cube.values[9] = fabs(max_pt.x - min_pt.x);

//	std::string sid = (int)min_pt.x;
//	sid += "cube_obj";
				name += "cube_obj";

//	viz.addText ((boost::format ("Score:     %s") % name.c_str()).str(), min_pt.x, min_pt.y, 20, 1.0, 1.0, 1.0, name.c_str());
				viz.addText3D((boost::format ("Score:     %s") % name.c_str()).str(), min_pt, 0.008);
				name += "cube";

        viz.addCube(coeff_cube, name.c_str());
  }

  void drawCube(pcl::visualization::PCLVisualizer& viz, pcl::PointXYZRGB min_pt, pcl::PointXYZRGB max_pt)
  {
	pcl::ModelCoefficients coeff_cube;

	coeff_cube.values.resize(10);
	coeff_cube.values[0] = min_pt.x + (max_pt.x - min_pt.x)/2;
	coeff_cube.values[1] = min_pt.y + (max_pt.y - min_pt.y)/2;
	coeff_cube.values[2] = min_pt.z + (max_pt.z - min_pt.z)/2;
	coeff_cube.values[3] = 0;
	coeff_cube.values[4] = 0;
	coeff_cube.values[5] = 0;
	coeff_cube.values[6] = 1;
	coeff_cube.values[7] = fabs(max_pt.x - min_pt.x);
	coeff_cube.values[8] = fabs(max_pt.x - min_pt.x);
	coeff_cube.values[9] = fabs(max_pt.x - min_pt.x);

	std::string sid = "cubex";
//	sid += (int)min_pt.x;

	viz.addCube(coeff_cube, sid.c_str());
  }
  
  void OpenCVInit()
  {
	cascadeName = "banana_all_nsym_20.xml";/*"training/training_glass_9s_nsym.xml";*/
	if( !cascade.load(cascadeName.c_str()) ) {
		cout << "Training File could not be loaded\n";
		return;
	}
  }

	BoundingBox LabelCluster2D(CloudPtr cloud)
	{
		double ret = 0.0;
		
		pcl::PointXYZRGB min_pt, max_pt;
    		pcl::getMinMax3D(cloud.operator*(), min_pt, max_pt);
		double ht = (max_pt.y - min_pt.y);
		double wid = (max_pt.x - min_pt.x);

		pcl::PointXYZ pt1 = GetNormalXYZCoord(min_pt.x, min_pt.y, min_pt.z);
		pcl::PointXYZ pt2 = GetNormalXYZCoord(max_pt.x, max_pt.y, min_pt.z);

		int height = abs(pt2.y - pt1.y);
		int width = abs(pt1.x - pt2.x);

		// refining
		double scale = 1.4;
		int deltaH = abs(floor(height*scale) - height);
		int deltaW = abs(floor(width*scale) - width);

		pt1.x = max(1.0f, pt1.x - (deltaW/2));
		pt2.x = min(638.0f, pt2.x + (deltaW/2));
		pt1.y = max(1.0f, pt1.y - (deltaH/2));
		pt2.y = min(478.0f, pt2.y + (deltaH/2));

		// get sub image
		cv::Mat subImage = m_currentImage(cv::Range(pt1.y, pt2.y), cv::Range(pt1.x, pt2.x));
	
		//cv::Mat smallImg( min(64, width), min(64, height), CV_8UC1);
		//cv::resize(imageRGB, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
		
		double prob = 0.0;
		cv::Rect obj_rect = m_pObjectDetector->testClassifyMultiScale(subImage, 16, prob);

		// change the rect to main coordinate frame
		obj_rect.x = (obj_rect.x + pt1.x);
		obj_rect.y = (obj_rect.y + pt1.y);

		pcl::PointXYZ minPt = GetPclXYZCoord(obj_rect.x, obj_rect.y, min_pt.z);
		pcl::PointXYZ maxPt = GetPclXYZCoord(obj_rect.x + obj_rect.width, obj_rect.y + obj_rect.height, min_pt.z);

		BoundingBox bb;
		bb.x = minPt.x;
		bb.y = minPt.y;
		bb.height = abs(maxPt.y - minPt.y);
		bb.width = abs(maxPt.x - minPt.x);
		bb.prob = prob;

		ret = prob;
		return bb;
	}

	void GetImage(const CloudConstPtr &cloud)
	{
		cv::Mat imageRGB = cv::Mat(480, 640, CV_8UC3);
	  for(unsigned int h = 0; h < cloud->height; h++) {
		  for(unsigned int w = 0; w < cloud->width; w++) {
			  imageRGB.at<cv::Vec3b>(h,w)[0] = cloud->points[h*cloud->width + w].b;
			  imageRGB.at<cv::Vec3b>(h,w)[1] = cloud->points[h*cloud->width + w].g;
			  imageRGB.at<cv::Vec3b>(h,w)[2] = cloud->points[h*cloud->width + w].r;
			}
	  }
		imageRGB.assignTo(m_currentImage);
	}

	pcl::PointXYZ GetNormalXYZCoord(double x, double y, double z)
	{
		float r_x = ((x * 525.0)/z) + 320;
		float r_y = ((y * 525.0)/z) + 240;

		pcl::PointXYZ ret;
		ret.x = r_x;
		ret.y = r_y;

		return ret;
	}

	pcl::PointXYZ GetPclXYZCoord(int x, int y, double ref_z)
	{
		pcl::PointXYZ pt;
		pt.x = ( ((x - 320.0) * ref_z) / 525.0 );
		pt.y = ( ((y - 240.0) * ref_z) / 525.0 );

		return pt;
	}

  vector<cv::Rect> DetectObject(const CloudConstPtr &cloud)
  {
					double scale = 1.2;
					vector<cv::Rect> objs;

					cvNamedWindow("result", CV_WINDOW_AUTOSIZE);

					cout << cloud->height << " " << cloud->width << endl;
					// get the image from the cloud
					cv::Mat imageRGB = cv::Mat(640, 480, CV_8UC3);
					for(unsigned int h = 0; h < cloud->height; h++) {
						for(unsigned int w = 0; w < cloud->width; w++) {
							imageRGB.at<cv::Vec3b>(h,w)[0] = cloud->points[h*cloud->width + w].b;
							imageRGB.at<cv::Vec3b>(h,w)[1] = cloud->points[h*cloud->width + w].g;
							imageRGB.at<cv::Vec3b>(h,w)[2] = cloud->points[h*cloud->width + w].r;
							}
					}
				//	imageRGB = imageRGB.t();
					IplImage *img = new IplImage(imageRGB);

					cv::Mat imgMat(img);
							
					cv::Mat gray, smallImg( cvRound(imgMat.rows/scale), cvRound(imgMat.cols/scale), CV_8UC1 );
					cv::cvtColor( imgMat, gray, CV_BGR2GRAY );
					cv::resize( gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR );
					cv::equalizeHist( smallImg, smallImg );

					cascade.detectMultiScale( smallImg, objs, scale, 2, 0|CV_HAAR_SCALE_IMAGE, cv::Size(30, 30) );

					// display the objs
					for( vector<cv::Rect>::const_iterator it = objs.begin(); it != objs.end(); it++) {
						cv::Point center;
						int radius;
					
						center.x = cvRound( (it->x + it->width*0.5) * scale);
						center.y = cvRound( (it->y + it->height*0.5) * scale);
						radius = cvRound( (it->width + it->height) * scale * 0.25);
						circle(imageRGB, center, radius, CV_RGB(255,0,255), 3, 8, 0);
					}


					cv::imshow("result", imageRGB);
					cv::waitKey(0);

				//	cvDestroyWindow("result");
					return objs;
  }

BoundingBox LabelCluster3D(CloudPtr objCloud)
{
	double prob = 0.0;

	pcl::PointCloud<pcl::PointXYZ>::Ptr changedXYZCloud (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*objCloud, *changedXYZCloud);

	vfh_model histogram;
	bool bFeatures = CalculateVFH(changedXYZCloud, histogram);

	if(bFeatures == false) return BoundingBox();

	nearestKSearch (*flannIndex_, histogram, K, k_indices_, k_distances_);

	int min_dist = 10000, min_i = -1;
	// Output the results on screen
	for (int i = 0; i < K; ++i) {
		pcl::console::print_info ("    %d - %s (%d) with a distance of: %f\n", i, models3d_.at (k_indices_[0][i]).first.c_str (), k_indices_[0][i], k_distances_[0][i]);
		if(min_dist > k_distances_[0][i]) {
			min_dist = k_distances_[0][i];
			min_i = i;
		}
	}

	if(/*min_dist > thresh ||*/ min_i < 0)
		return BoundingBox();

	std::string fObj = models3d_.at(k_indices_[0][min_i]).first.c_str();

	if( fObj.find(m_objectName) != std::string::npos ) {
		int dist = k_distances_[0][min_i];
		int bin = ceil((dist*1.0)/30.0);
		prob = (1.0 / (bin*1.0));//(dist == 0 ? 0.0001 : dist*1.0));
	}

	pcl::PointXYZRGB min_pt, max_pt;
	pcl::getMinMax3D(objCloud.operator*(), min_pt, max_pt);
	double ht = (max_pt.y - min_pt.y);
	double wid = (max_pt.x - min_pt.x);

	BoundingBox bb;
	bb.x = min_pt.x;
	bb.y = min_pt.y;
	bb.z = min_pt.z;
	bb.height = abs(max_pt.y - min_pt.y);
	bb.width = abs(max_pt.x - min_pt.x);
	bb.prob = prob;

	return bb;
}

bool CalculateFPFH(pcl::PointCloud<pcl::PointXYZ>::Ptr objCloud, vfh_model &vfh)
{
        try {
                // Normal Estimation over whole cloud (dataset)
                pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
                ne.setInputCloud(objCloud);

                pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
                ne.setSearchMethod(tree);

                pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

                ne.setRadiusSearch(0.03);
                ne.compute(*cloud_normals);

                //VFH Estimation
                pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> vfhE;
                vfhE.setInputCloud(objCloud);
                vfhE.setInputNormals(cloud_normals);

                pcl::search::KdTree<pcl::PointXYZ>::Ptr treeVfh (new pcl::search::KdTree<pcl::PointXYZ> ());
                vfhE.setSearchMethod(treeVfh);
                vfhE.setRadiusSearch(0.03);

                pcl::PointCloud<pcl::FPFHSignature33>::Ptr vfhs (new pcl::PointCloud<pcl::FPFHSignature33>());
                vfhE.compute(*vfhs);

                // store vfh features
                vfh.second.resize(33);

                for (size_t i = 0; i < 33; i++) {
                        vfh.second[i] = (*vfhs).points[0].histogram[i];
                }
//                vfh.first = path.string();
        }
        catch (pcl::InvalidConversionException e) {
                return false;
        }

        return true;
}

bool CalculateVFH(pcl::PointCloud<pcl::PointXYZ>::Ptr objCloud, vfh_model &vfh)
{
        try {
		// Normal Estimation over whole cloud (dataset)
                pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
                ne.setInputCloud(objCloud);

        //      pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree (new pcl::KdTreeFLANN<pcl::PointXYZ>());
                pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>());
                ne.setSearchMethod(tree);

                pcl::PointCloud<pcl::Normal>::Ptr cloud_normals (new pcl::PointCloud<pcl::Normal>);

                ne.setRadiusSearch(0.03);
                ne.compute(*cloud_normals);

                //VFH Estimation
                pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfhE;
                vfhE.setInputCloud(objCloud);
                vfhE.setInputNormals(cloud_normals);

        //      pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr treeVfh (new pcl::KdTreeFLANN<pcl::PointXYZ> ());
                pcl::search::KdTree<pcl::PointXYZ>::Ptr treeVfh (new pcl::search::KdTree<pcl::PointXYZ> ());
                vfhE.setSearchMethod(treeVfh);

                pcl::PointCloud<pcl::VFHSignature308>::Ptr vfhs (new pcl::PointCloud<pcl::VFHSignature308>());
                vfhE.compute(*vfhs);

                // store vfh features
                vfh.second.resize(308);

                for (size_t i = 0; i < 308; i++) {
                        vfh.second[i] = (*vfhs).points[0].histogram[i];
                }

        }
        catch (pcl::InvalidConversionException e) {
                return false;
        }

        return true;
}

inline void
nearestKSearch (flann::Index<flann::ChiSquareDistance<float> > &index, const vfh_model &model,
                int k, flann::Matrix<int> &indices, flann::Matrix<float> &distances)
{
  // Query point
  flann::Matrix<float> p = flann::Matrix<float>(new float[model.second.size ()], 1, model.second.size ());
  memcpy (&p.ptr ()[0], &model.second[0], p.cols * p.rows * sizeof (float));

  indices = flann::Matrix<int>(new int[k], 1, k);
  distances = flann::Matrix<float>(new float[k], 1, k);
  index.knnSearch (p, indices, distances, k, flann::SearchParams (512));
  delete[] p.ptr ();
}

bool
loadFileList (std::vector<vfh_model> &models, const std::string &filename)
{
  ifstream fs;
  fs.open (filename.c_str ());
  if (!fs.is_open () || fs.fail ())
    return (false);

  std::string line;
  while (!fs.eof ())
  {
    getline (fs, line);
    if (line.empty ())
      continue;
    vfh_model m;
    m.first = line;
    models.push_back (m);
  }
  fs.close ();
  return (true);
}

int Init3DObjDetector()
{
	K = 6;
	thresh = 30;

	std::string kdtree_idx_file_name = m_objectName + "_kdtree.idx";
  	std::string training_data_h5_file_name = m_objectName + "_training_data.h5";
  	std::string training_data_list_file_name = m_objectName + "_training_data.list";

	// Check if the data has already been saved to disk
	if (!boost::filesystem::exists (training_data_h5_file_name.c_str()) || !boost::filesystem::exists (training_data_list_file_name.c_str()))
	{
	  	pcl::console::print_error ("Could not find training data models files %s and %s!\n",
		training_data_h5_file_name.c_str (), training_data_list_file_name.c_str ());
	    	return (-1);
	}
	else
	{
		loadFileList (models3d_, training_data_list_file_name);
	    	flann::load_from_file (pcdTrainingData_, training_data_h5_file_name, "training_data");
	    	pcl::console::print_highlight ("Training data found. Loaded %d FPFH models from %s/%s.\n",
		(int)pcdTrainingData_.rows, training_data_h5_file_name.c_str (), training_data_list_file_name.c_str ());
	}

	// Check if the tree index has already been saved to disk
	if (!boost::filesystem::exists (kdtree_idx_file_name))
	{
		pcl::console::print_error ("Could not find kd-tree index in file %s!", kdtree_idx_file_name.c_str ());
	    	return (-1);
	}
	else
	{
  	flann::Index<flann::ChiSquareDistance<float> > *pindex = new flann::Index<flann::ChiSquareDistance<float> > (pcdTrainingData_, flann::SavedIndexParams (kdtree_idx_file_name.c_str()));//"kdtree.idx"));
		pindex->buildIndex ();
		flannIndex_ = pindex;
	}
	return 0;

}

void SetObjectName(std::string name) {
	m_objectName = name;
}

  pcl::PointXYZ		m_predictedPoint;
  Kalman		m_kalman;

	std::string							m_objectName;
	cv::Mat									m_currentImage;
	ObjectDetection*				m_pObjectDetector;

  int K;
  int thresh;
  flann::Index<flann::ChiSquareDistance<float> > *flannIndex_;
  std::vector<vfh_model> models3d_;
  flann::Matrix<int> k_indices_;
  flann::Matrix<float> k_distances_;
  flann::Matrix<float> pcdTrainingData_;
  vector<Clusters> segmentedClusters_;

  vector<CirclePoints> detectedObjsXYZ_;
  pcl::visualization::CloudViewer viewer_;
  pcl::PointCloud<pcl::Normal>::Ptr normals_;
  CloudPtr curr_cloud_;
  CloudPtr cloud_pass_;
  CloudPtr target_cloud_;
  CloudPtr cloud_pass_downsampled_;
  CloudPtr plane_cloud_;
  CloudPtr nonplane_cloud_;
  CloudPtr cloud_hull_;
  CloudPtr segmented_cloud_;
  CloudPtr reference_;
  std::vector<pcl::Vertices> hull_vertices_;
  
  std::string device_id_;
  boost::mutex mtx_;
  bool new_cloud_;
  pcl::NormalEstimationOMP<PointType, pcl::Normal> ne_; // to store threadpool
  boost::shared_ptr<ParticleFilter> tracker_;
  int counter_;
  bool use_convex_hull_;
  bool visualize_non_downsample_;
  bool visualize_particles_;
  double tracking_time_;
  double computation_time_;
  double downsampling_time_;
  double downsampling_grid_size_;


  // OpenCV params
  cv::CascadeClassifier cascade;
  string cascadeName;
};

int
main (int argc, char** argv)
{
  bool use_convex_hull = true;
  bool visualize_non_downsample = false;
  bool visualize_particles = true;
  bool use_fixed = false;

  double downsampling_grid_size = 0.01;
  
  if (pcl::console::find_argument (argc, argv, "-C") > 0)
    use_convex_hull = false;
  if (pcl::console::find_argument (argc, argv, "-D") > 0)
    visualize_non_downsample = true;
  if (pcl::console::find_argument (argc, argv, "-P") > 0)
    visualize_particles = false;
  if (pcl::console::find_argument (argc, argv, "-fixed") > 0)
    use_fixed = true;
  pcl::console::parse_argument (argc, argv, "-d", downsampling_grid_size);
  if (argc < 2)
  {
    exit (1);
  }
  
  std::string device_id = std::string (argv[1]);

  if (device_id == "--help" || device_id == "-h")
  {
    exit (1);
  }

	std::string objName = argv[3];
  
  // open kinect
  TestSegmentTracking<pcl::PointXYZRGB> v (device_id, 8, downsampling_grid_size,
                                             use_convex_hull,
                                             visualize_non_downsample, visualize_particles,
                                             use_fixed, objName);
  v.run ();
}

#include <iostream>

#include <pcl/ModelCoefficients.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl_conversions/pcl_conversions.h>

#include <cmath>
#include <vector>

#include "geometry_msgs/msg/point_stamped.hpp"
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2/tf2/convert.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include "visualization_msgs/visualization_msgs/msg/marker.hpp"

rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr planes_pub;
rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cylinder_pub;
rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;

std::shared_ptr<rclcpp::Node> node;
std::shared_ptr<tf2_ros::TransformListener> tf_listener_{nullptr};
std::unique_ptr<tf2_ros::Buffer> tf_buffer_;

typedef pcl::PointXYZRGB PointT;

int marker_id = 0;
float error_margin = 0.02;  // 2 cm margin for error
float target_radius = 0.11;
bool verbose = false;
bool detect_cylinder = false;

std::vector<std::string> color_labels;

class ColorClassifier {
    private:
        std::vector<std::vector<int>> rgb_values = {
            {255, 0, 0},    // Red
            {204, 71, 65},
            {130, 120, 120},
            {116, 69, 68},
            {147, 53, 49},
            {154, 143, 142},
            {62, 37, 36},
            {96, 33, 30},
            {134, 134, 134},
            {159, 153, 152},
            {137, 105, 105},
            {190, 101, 100},
            {0, 255, 0},    // Green
            {63, 137, 62},
            {25, 124, 23},
            {128, 136, 128},
            {158, 165, 157},
            {139, 150, 139},
            {117, 131, 116},
            {102, 132, 98},
            {126, 146, 125},
            {80, 93, 79},
            {46, 92, 45},
            {84, 110, 84},
            {0, 0, 255},    // Blue
            {38, 62, 84},
            {26, 42, 58},
            {124, 129, 133},
            {79, 119, 155},
            {115, 121, 131},
            {123, 149, 173},
            {150, 152, 154},
            {145, 161, 177},
            {78, 69, 90},
            {44, 59, 77},
            {255, 255, 0},  // Yellow
            {101, 87, 0},
            {170, 170, 170},  // Gray
            {100, 100, 100},
            {120, 120, 120},
            {79, 80, 80},
            {77, 78, 78},
            {159, 161, 160},
            {147, 145, 135},
            {97, 97, 97},
            {0,0, 0} // Black
        };

        std::vector<std::string> color_labels = {"red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red", "red",
                                        "green", "green", "green", "green", "green", "green", "green", "green", "green", "green", "green", "green",
                                        "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue", "blue",  "blue", "blue",
                                        "yellow", "yellow",
                                        "gray", "gray", "gray", "gray", "gray", "gray", "gray", "gray",
                                        "black"};

    public:
        void trainClassifier() {
            std::cout << "Training Classifier..." << std::endl;
            for (size_t i = 0; i < rgb_values.size(); ++i) {
                std::cout << color_labels[i] << ": ";
                for (int j = 0; j < 3; ++j) {
                    std::cout << rgb_values[i][j] << " ";
                }
                std::cout << std::endl;
            }
        }

        std::string predictColor(std::vector<int> rgb) {
            double minDist = INT_MAX;
            std::string predictedColor;
            for (size_t i = 0; i < rgb_values.size(); ++i) {
                double dist = 0;
                for (int j = 0; j < 3; ++j) {
                    dist += pow(rgb[j] - rgb_values[i][j], 2);
                }
                dist = sqrt(dist);
                if (dist < minDist) {
                    minDist = dist;
                    predictedColor = color_labels[i];
                }
            }
            return predictedColor;
        }
};

ColorClassifier colorClassifier;

void when_to_detect_cylinder_cb(const std_msgs::msg::Bool::SharedPtr msg) {
    detect_cylinder = true;
    std::cout << "READY TO DETECT" << std::endl;
}

// set up PCL RANSAC objects

void cloud_cb(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
    if (!detect_cylinder) {
        return;
    }
    
    // save timestamp from message
    rclcpp::Time now = (*msg).header.stamp;

    // set up PCL objects
    pcl::PassThrough<PointT> pass;
    pcl::NormalEstimation<PointT, pcl::Normal> ne;
    pcl::SACSegmentationFromNormals<PointT, pcl::Normal> seg;
    pcl::PCDWriter writer;
    pcl::ExtractIndices<PointT> extract;
    pcl::ExtractIndices<pcl::Normal> extract_normals;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT>());
    Eigen::Vector4f centroid;

    // set up pointers
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>);
    pcl::PCLPointCloud2::Ptr pcl_pc(new pcl::PCLPointCloud2);
    pcl::PointCloud<PointT>::Ptr cloud_filtered(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
    pcl::PointCloud<PointT>::Ptr cloud_filtered2(new pcl::PointCloud<PointT>);
    pcl::PointCloud<pcl::Normal>::Ptr cloud_normals2(new pcl::PointCloud<pcl::Normal>);
    pcl::ModelCoefficients::Ptr coefficients_plane(new pcl::ModelCoefficients), coefficients_cylinder(new pcl::ModelCoefficients);
    pcl::PointIndices::Ptr inliers_plane(new pcl::PointIndices), inliers_cylinder(new pcl::PointIndices);
    pcl::PointCloud<PointT>::Ptr cloud_plane(new pcl::PointCloud<PointT>());

    // convert ROS msg to PointCloud2
    pcl_conversions::toPCL(*msg, *pcl_pc);

    // convert PointCloud2 to templated PointCloud
    pcl::fromPCLPointCloud2(*pcl_pc, *cloud);

    if (verbose) {
        std::cerr << "PointCloud has: " << cloud->points.size() << " data points." << std::endl;
    }

    // Build a passthrough filter to remove spurious NaNs
    pass.setInputCloud(cloud);
    pass.setFilterFieldName("x");
    pass.setFilterLimits(0, 10);
    pass.filter(*cloud_filtered);
    if (verbose) {
        std::cerr << "PointCloud after filtering has: " << cloud_filtered->points.size() << " data points." << std::endl;
    }

    // Estimate point normals
    ne.setSearchMethod(tree);
    ne.setInputCloud(cloud_filtered);
    ne.setKSearch(50);
    ne.compute(*cloud_normals);

    // Create the segmentation object for the planar model and set all the
    // parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_NORMAL_PLANE);
    seg.setNormalDistanceWeight(0.1);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.03);
    seg.setInputCloud(cloud_filtered);
    seg.setInputNormals(cloud_normals);

    seg.segment(*inliers_plane, *coefficients_plane);
    if (verbose) {
        std::cerr << "Plane coefficients: " << *coefficients_plane << std::endl;
    }

    // Extract the planar inliers from the input cloud
    extract.setInputCloud(cloud_filtered);
    extract.setIndices(inliers_plane);
    extract.setNegative(false);
    extract.filter(*cloud_plane);

    // Remove the planar inliers, extract the rest
    extract.setNegative(true);
    extract.filter(*cloud_filtered2);
    extract_normals.setNegative(true);
    extract_normals.setInputCloud(cloud_normals);
    extract_normals.setIndices(inliers_plane);
    extract_normals.filter(*cloud_normals2);

    // Create the segmentation object for cylinder segmentation and set all the
    // parameters
    seg.setOptimizeCoefficients(true);
    seg.setModelType(pcl::SACMODEL_CYLINDER);
    seg.setMethodType(pcl::SAC_RANSAC);
    seg.setNormalDistanceWeight(0.1);
    seg.setMaxIterations(100);
    seg.setDistanceThreshold(0.05);
    seg.setRadiusLimits(0.06, 0.2);
    seg.setInputCloud(cloud_filtered2);
    seg.setInputNormals(cloud_normals2);

    // Obtain the cylinder inliers and coefficients
    seg.segment(*inliers_cylinder, *coefficients_cylinder);

    // return if no cylinder was detected
    int coef_size = (*coefficients_cylinder).values.size();
    if (coef_size == 0) {
        return;
    }

    if (verbose) {
        std::cerr << "Cylinder coefficients: " << *coefficients_cylinder << std::endl;
    }

    float detected_radius = (*coefficients_cylinder).values[6];

    if (std::abs(detected_radius - target_radius) > error_margin) {
        return;
    }

    // extract cylinder
    extract.setInputCloud(cloud_filtered2);
    extract.setIndices(inliers_cylinder);
    extract.setNegative(false);
    pcl::PointCloud<PointT>::Ptr cloud_cylinder(new pcl::PointCloud<PointT>());
    extract.filter(*cloud_cylinder);
	
	pcl::compute3DCentroid(*cloud_cylinder, centroid);
	// Get the index of the centroid point
	int centroid_index = -1;
	float min_distance = std::numeric_limits<float>::max();
	for (size_t i = 0; i < cloud_cylinder->points.size(); ++i) {
		float distance = pow(cloud_cylinder->points[i].x - centroid[0], 2) +
		                 pow(cloud_cylinder->points[i].y - centroid[1], 2) +
		                 pow(cloud_cylinder->points[i].z - centroid[2], 2);
		if (distance < min_distance) {
		    min_distance = distance;
		    centroid_index = i;
		}
	}

	// Access the RGB components at the centroid index
	int cylinder_r = cloud_cylinder->points[centroid_index].r;
	int cylinder_g = cloud_cylinder->points[centroid_index].g;
	int cylinder_b = cloud_cylinder->points[centroid_index].b;

	// Print out the centroid value and RGB components
	if (centroid_index != -1) {
		std::cout << "Centroid: " << cloud_cylinder->points[centroid_index].x << " "
		          << cloud_cylinder->points[centroid_index].y << " "
		          << cloud_cylinder->points[centroid_index].z << std::endl;
		std::cout << "RGB: " << cylinder_r << " " << cylinder_g << " " << cylinder_b << std::endl;
		} else {
			std::cout << "Centroid not found!" << std::endl;
	}

    // predict color
    std::vector<int> rgb = {cylinder_r, cylinder_g, cylinder_b};
    std::string predictedColor = colorClassifier.predictColor(rgb);
    std::cout << "Predicted color: " << predictedColor << std::endl;

    if (predictedColor == "gray") {
        std::cout << "TOLE NI CILINDER" << std::endl;
        return;
    }

    // print color labels
    std::cout << "Color labels: ";
    for (size_t i = 0; i < color_labels.size(); ++i) {
        std::cout << color_labels[i] << " ";
    }
    std::cout << std::endl;

    for (const auto& label : color_labels) {
        if (label == predictedColor) {
            return;
        }
    }

    //make marker color based on predicted color
    if (predictedColor == "red") {
        cylinder_r = 255;
        cylinder_g = 0;
        cylinder_b = 0;
    } else if (predictedColor == "green") {
        cylinder_r = 0;
        cylinder_g = 255;
        cylinder_b = 0;
    } else if (predictedColor == "blue") {
        cylinder_r = 0;
        cylinder_g = 0;
        cylinder_b = 255;
    } else if (predictedColor == "yellow") {
        cylinder_r = 255;
        cylinder_g = 255;
        cylinder_b = 0;
    } else if (predictedColor == "black") {
        cylinder_r = 0;
        cylinder_g = 0;
        cylinder_b = 0;
    }


    // calculate marker
    // pcl::compute3DCentroid(*cloud_cylinder, centroid);
    if (verbose) {
        std::cerr << "centroid of the cylindrical component: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << " " << centroid[3] << std::endl;
    }

    geometry_msgs::msg::PointStamped point_camera;
    geometry_msgs::msg::PointStamped point_map;
    visualization_msgs::msg::Marker marker;
    geometry_msgs::msg::TransformStamped tss;

    // set up marker messages
    std::string toFrameRel = "map";
    std::string fromFrameRel = (*msg).header.frame_id;
    point_camera.header.frame_id = fromFrameRel;

    point_camera.header.stamp = now;
    point_camera.point.x = centroid[0];
    point_camera.point.y = centroid[1];
    point_camera.point.z = centroid[2];

    try {
        tss = tf_buffer_->lookupTransform(toFrameRel, fromFrameRel, now);
        tf2::doTransform(point_camera, point_map, tss);
    } catch (tf2::TransformException& ex) {
        std::cout << ex.what() << std::endl;
    }

    if (verbose) {
        std::cerr << "point_camera: " << point_camera.point.x << " " << point_camera.point.y << " " << point_camera.point.z << std::endl;
        std::cerr << "point_map: " << point_map.point.x << " " << point_map.point.y << " " << point_map.point.z << std::endl;
    }

    // publish marker
    marker.header.frame_id = "map";
    marker.header.stamp = now;

    marker.ns = "cylinder";
    // marker.id = 0; // only latest marker
    marker.id = marker_id++;  // generate new markers

    marker.type = visualization_msgs::msg::Marker::CYLINDER;
    marker.action = visualization_msgs::msg::Marker::ADD;

    marker.pose.position.x = point_map.point.x;
    marker.pose.position.y = point_map.point.y;
    marker.pose.position.z = point_map.point.z;
    marker.pose.orientation.x = 0.0;
    marker.pose.orientation.y = 0.0;
    marker.pose.orientation.z = 0.0;
    marker.pose.orientation.w = 1.0;

    marker.scale.x = 0.1;
    marker.scale.y = 0.1;
    marker.scale.z = 0.1;

    marker.color.r = cylinder_r / 255.0f;
    marker.color.g = cylinder_g / 255.0f;
    marker.color.b = cylinder_b / 255.0f;
    marker.color.a = 1.0f;

    // marker.lifetime = rclcpp::Duration(1,0);
    //marker.lifetime = rclcpp::Duration(10, 0);

    marker_pub->publish(marker);

    //////////////////////////// publish result point clouds /////////////////////////////////

    // convert to pointcloud2, then to ROS2 message
    sensor_msgs::msg::PointCloud2 plane_out_msg;
    pcl::PCLPointCloud2::Ptr outcloud_plane(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*cloud_plane, *outcloud_plane);
    pcl_conversions::fromPCL(*outcloud_plane, plane_out_msg);
    planes_pub->publish(plane_out_msg);

    // publish cylinder
    sensor_msgs::msg::PointCloud2 cylinder_out_msg;
    pcl::PCLPointCloud2::Ptr outcloud_cylinder(new pcl::PCLPointCloud2());
    pcl::toPCLPointCloud2(*cloud_cylinder, *outcloud_cylinder);
    pcl_conversions::fromPCL(*outcloud_cylinder, cylinder_out_msg);
    cylinder_pub->publish(cylinder_out_msg);

    color_labels.push_back(predictedColor);
}


int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    std::cout << "cylinder_segmentation" << std::endl;

    node = rclcpp::Node::make_shared("cylinder_segmentation");

    // create subscriber
    node->declare_parameter<std::string>("topic_pointcloud_in", "/oakd/rgb/preview/depth/points");
    std::string param_topic_pointcloud_in = node->get_parameter("topic_pointcloud_in").as_string();
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription = node->create_subscription<sensor_msgs::msg::PointCloud2>(param_topic_pointcloud_in, 10, &cloud_cb);

    // create subscriber for when to detect
    node->declare_parameter<bool>("detect_cylinder", false);
    detect_cylinder = node->get_parameter("detect_cylinder").as_bool();
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription2 = node->create_subscription<std_msgs::msg::Bool>("/when_to_detected_cylinder", 10, &when_to_detect_cylinder_cb);
    
    // setup tf listener
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    // create publishers
    planes_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("planes", 1);
    cylinder_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("cylinder", 1);
    marker_pub = node->create_publisher<visualization_msgs::msg::Marker>("detected_cylinder", 1);

    // color classifier
    colorClassifier.trainClassifier();

    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
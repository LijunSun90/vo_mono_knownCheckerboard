// vo_mono_knownCheckerboard.cpp
// Calling convention:
// ./vo_mono_knownCheckerboard 7 8 chessboardsLijun5.txt 25.4 0.5 

// -------------------------------------------------------------------
// Calculate the trajectory (Visual Odometry) of the monocular camera.
//
// STEP 1: Load the intrinsics and distortion coefficients.
// STEP 2: Find the target object and the corresponding corners.
// STEP 3: Pose estimation of the target object.
// STEP 4: Transfer to the pose estimation of the monocular camera.
// STEP 5: Calculate the trajectory (VO) of the monocular camera.
//
// Step 6: Save the poses. -- This the output what we want!!!
//
// -------------------------------------------------------------------

#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;

int flag_found = 0;

//
void help(char* argv[]){
printf("\n\n"
" Calling convention:\n"
"\n"
"   vo_mono_knownCheckerboard  board_w  board_h  image_list\n"
"\n"
"   or\n"
"\n"
"   vo_mono_knownCheckerboard  board_w  board_h  image_list object_size\n"
"\n"
"   or\n"
"\n"
"   vo_mono_knownCheckerboard  board_w  board_h  image_list object_size image_sf \n"
"\n"
" WHERE:\n\n"
" board_w, board_h --the number of corners along the row and columns respectively\n"
" image_list       --space separated list of path/filename of checkerboard images\n"
" object_size      --the physical size of the grid of the chessboard\n"
" image_sf         --the scale factor of the input image used in image processing\n"
"\n"
" \nFor example:\n"
" ./vo_mono_knownCheckerboard 7 8 chessboardsLijun5.txt 25.4 0.5\n"
" Hit ‘p’ to pause/unpause, ESC to quit\n\n");
}

int main(int argc, char* argv[]){
    // --------------------------------------------------
    // Parameters initilization.
    // --------------------------------------------------
    int n_boards = 0; // Will be set by input list.
    float object_size = 25.4; // or 25.4 mm
    float image_sf = 0.5f;
    float delay = 1.f;
    int board_w = 0;
    int board_h = 0;
    // The scale variable here is only used for trajectory display.
    double scale = 0.3;

    if (argc < 3 || argc > 6){
        cout << "\nERROR: Wrong number of input parameters";
        help(argv);
        return -1;
    }
    board_w = atoi(argv[1]);
    board_h = atoi(argv[2]);
    ifstream inImageNames(argv[3]);
    if(argc > 3) object_size = atof(argv[4]);
    if(argc > 4) image_sf = atof(argv[5]);

    int board_n = board_w * board_h;
    cv::Size board_sz = cv::Size(board_w, board_h);

    string names;
    while(getline(inImageNames, names)) {
        n_boards ++;
        // cout << names << endl;
        // cout << n_boards << endl;
    }
    inImageNames.clear();
    inImageNames.seekg(ios::beg);

    // ----------------------------------------------------------
    // STEP 1: Load the intrinsics and distortion coefficients.
    // ----------------------------------------------------------
    //
    cv::FileStorage fs("intrinsics.xml", cv::FileStorage::READ);
    cout << "\nimage width: " << (int)fs["image_width"];
    cout << "\nimage height: " << (int)fs["image_height"];

    cv::Mat intrinsic_matrix_loaded, distorition_coeffs_loaded;
    fs["camera_matrix"] >> intrinsic_matrix_loaded;
    fs["distortion_coefficients"] >> distorition_coeffs_loaded;

    cout << "\n\nintrinsic matrix: \n" << intrinsic_matrix_loaded;
    cout << "\n\ndistortion coefficients: \n" << distorition_coeffs_loaded << endl;
    fs.release();

    // ALLOCATE STORAGE.
    //
    // The coordinates of image points p in the camera coordinate system 
    // centered on the camera.
    vector< vector<cv::Point2f> > image_points;
    // The coordinates of object points P in the coordiante system 
    // centered on the object.
    vector< vector<cv::Point3f> > object_points;

    // The point p is related to point P by applying a rotation matrix R
    // and a translation vector t to P.
    // Variables use for storage.
    vector< cv::Mat > rvecs, tvecs;
    cv::Mat rvec, tvec, rMat;
    // Use for display trajectory.
    cv::Mat trajectory_Panel = cv::Mat(600, 600, CV_8UC3, cv::Scalar::all(0));
    cv::Mat trajectory = cv::Mat::zeros(cv::Size(1, 3), CV_64F);
    cv::Mat trajectoryPrev;

    cv::Mat object_position = cv::Mat::zeros(cv::Size(1, 3), CV_64F);
    // Let the object_position located at the center of the chessboard.
    object_position.at<double>(0) = (board_w - 1) / 2 * object_size;
    object_position.at<double>(1) = (board_h - 1) / 2 * object_size;
    object_position.at<double>(2) = 0;    
    cv::Mat origin = cv::Mat::zeros(cv::Size(1, 3), CV_64F);
    origin.at<double>(0) = 300;
    origin.at<double>(1) = 300;
    origin.at<double>(2) = 300;

    // Capture corner views: loop until we've got n_boards successful
    // captures (all corners on the board are found).
    //
    cv::Size image_size;
    int successes = 0;
    // n_boards
    for(int frame=0; frame < n_boards; frame++){
        cv::Mat image0, image;
        getline(inImageNames, names);
        image0 = cv::imread(names);
        image_size = image0.size();
        cv::resize(image0, image, cv::Size(), image_sf, image_sf, cv::INTER_LINEAR);
        // --------------------------------------------------
        // STEP 2: Find the target object and the 
        //         corresponding corners.
        // --------------------------------------------------  
        //      
        // Find the board.
        // corners: Locations of the located corners in sub-pixel coordiantes.
        vector<cv::Point2f> corners;
        bool found = cv::findChessboardCorners(image, board_sz, corners);     

        // Draw it.
        //
        drawChessboardCorners(image, board_sz, corners, found);
        cv::imshow("Found keypoints", image);   // show in color f we did collect the image.
        cv::moveWindow("Found keypoints", 900, 0);

        // If we got a good board, add it to our data.
        //
        if (found){
            // image ^= cv::Scalar::all(255);

            // scale the coordinates back.
            cv::Mat mcorners(corners); // do not copy the data.
            mcorners *= (1./image_sf); // scale the corner coordinates.
            image_points.push_back(mcorners);

            object_points.push_back(vector<cv::Point3f>());
            vector<cv::Point3f>& opts = object_points.back();
            opts.resize(board_n);
            for(int j=0; j<board_n; j++){
                opts[j] = cv::Point3f((float)(j/board_w), (float)(j%board_w), 0.f) * object_size;
            }
            // cout << "Collected our " << (int)image_points.size() <<
            // " of " << n_boards << " needed chessboard images\n" << endl;

            // --------------------------------------------------
            // STEP 3: Pose estimation of the target object.
            // --------------------------------------------------
            //
            cv::solvePnPRansac(
                opts,
                corners,
                intrinsic_matrix_loaded,
                distorition_coeffs_loaded,
                rvec,
                tvec
            );
            // ----------------------------------------------------------------
            // STEP 4: Transfer to the pose estimation of the monocular camera.
            // ----------------------------------------------------------------
            //     
            // The pose estimation of the monocular camera, multiply with -1.
            // Otherwise, it is the pose estimation of the chessboard object.
            tvec = -1 * tvec;
            rvec = -1 * rvec;
            // cout << "rvec: \n" << rvec 
            cout << "\ntvec: \n" << tvec << endl;

            // Record the relatve rotation and translation.
            rvecs.push_back(rvec);
            tvecs.push_back(tvec);

            // ---------------------------------------------------------
            // STEP 5: Calculate the trajectory of the monocula// ----------------------------------------------------------------r camera.
            // ---------------------------------------------------------
            //  
            // Transform to the rotation matrix.
            cv::Rodrigues(rvec, rMat, cv::noArray());
            //
            trajectory = rMat * object_position + scale * tvec + origin;
            if (flag_found == 0){
                flag_found = 1;
                trajectoryPrev = trajectory;
            }

            cout << "\ntrajectory: \n" << trajectory 
            <<  "\ntrajectoryPrev: \n" << trajectoryPrev << endl;

            // Show the trajectory.
            cv::line(
                trajectory_Panel, 
                cv::Point(int(trajectoryPrev.at<double>(0)), int(trajectoryPrev.at<double>(1))),
                cv::Point(int(trajectory.at<double>(0)), int(trajectory.at<double>(1))),
                CV_RGB(0, 255, 0),
                2,
                cv::LINE_AA);
            cv::imshow("Trajectory of our camera - Press ESC to exit", trajectory_Panel);
            cv::moveWindow("Trajectory of our camera - Press ESC to exit", 0, 0);

            // Update.
            trajectoryPrev = trajectory.clone();
        }

        if((cv::waitKey(0) & 255) == 27) return -1;
    } // END COLLECTION FOR LOOP
    inImageNames.clear();
    inImageNames.seekg(ios::beg);

    cv::destroyAllWindows();

    // --------------------------------------------------
    // Step 6: Save the poses.
    // --------------------------------------------------
    //
    cout << "*** DONE! ***\n\n" 
    << "\nStoring poses.xml file...\n\n" << endl;
    fs.open("poses.xml", cv::FileStorage::WRITE);
    fs << "rvecs" << rvecs
    << "tvecs" << tvecs;
    fs.release();


    return 0;
}
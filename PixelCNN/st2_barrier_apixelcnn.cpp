#include <iostream>
#include <vector>
#include <thread>
#include <string>
#include <chrono>
#include <omp.h>
#include <unistd.h>


// finds elements that we need to calculate one node. stores in two vectors. one for the i positions one for j positions for example for kernel = 3, if center of the kernel is [2,2] this function will fill i positions with [1,1,1], and j positions with [1,2,3]
void findElements(int centeri, int centerj, int k, int maxw, std::vector<long double>& ei, std::vector<long double>& ej){
    
    int half = k/2; 
    // iterate over necessary rows
    for (int i = centeri-half; i < centeri; i++){
        // check if i will exceed the limits of the matrix
        if (i < 0)
            continue;
        // iterate over necessary columns
        for (int j = centerj-half; j < centerj+half+1; j++){
            // check if j will exceed the limits of the matrix
            if (j >= maxw | j < 0)
                continue;
            // push back to the referenced vectors for i positions and j positions
            ei.push_back(i);
            ej.push_back(j);
        }
    }
}


// incase we implement parallelism strat 2  
void calculateColumn(){

}

int main(int argc, char* argv[]){

    // initialize kernel, height, width
    int kernel=3;
	int height=10;
	int width=10;

    for(unsigned int i=1; i<argc; i++){
		std::string arg=argv[i];
        // check if argument for kernel is given
		if(arg=="-k"){
			if(argc>=i+1){
				try{
					kernel=std::stoull(argv[i+1]);
				}catch(std::invalid_argument){
					std::cerr << "Unable to interpret '" << argv[i+1] << "' as an integer" << std::endl;
					return 1;
				}
				i++;
			}
			else{
				std::cerr << "Missing value for kernel size after '-k'" << std::endl;
				return 1;
			}
			
		}
        // check if argument for matrix height is given
		else if (arg == "-h"){
			if(argc>=i+1){
				try{
					height=std::stoull(argv[i+1]);
				}catch(std::invalid_argument){
					std::cerr << "Unable to interpret '" << argv[i+1] << "' as an integer" << std::endl;
					return 1;
				}
				i++;
			}
			else{
				std::cerr << "Missing value for matrix height after '-h'" << std::endl;
				return 1;
			}
			
		}
        // check if argument for matrix width is given
		else if(arg=="-w"){
			if(argc>=i+1){
				try{
					width=std::stoull(argv[i+1]);
				}catch(std::invalid_argument){
					std::cerr << "Unable to interpret '" << argv[i+1] << "' as an integer" << std::endl;
					return 1;
				}
				i++;
			}
			else{
				std::cerr << "Missing value for matrix width after '-w'" << std::endl;
				return 1;
			}
		}
		// check if argument for matrix height and width is given
		else if(arg=="-hw"){
			if(argc>=i+1){
				try{
					height=std::stoull(argv[i+1]);
					width=std::stoull(argv[i+1]);
				}catch(std::invalid_argument){
					std::cerr << "Unable to interpret '" << argv[i+1] << "' as an integer" << std::endl;
					return 1;
				}
				i++;
			}
			else{
				std::cerr << "Missing value for matrix height and width after '-hw'" << std::endl;
				return 1;
			}
		}
	}

    auto start = std::chrono::high_resolution_clock::now();

    int header = kernel/2;

    // create matrix 0f 1s for the first few rows
    std::vector<std::vector<long double>> matrix(header, std::vector<long double>(width, 1));
    // create matrix of 0s to be calculated
    std::vector<std::vector<long double>> matrixBody(height-header, std::vector<long double>(width, 0));

    // concatenate the two matrices
    matrix.insert( matrix.end(), matrixBody.begin(), matrixBody.end());
	omp_set_nested(1);
    // iterate over elements of the matrix that need to be calculated 
	#pragma omp parallel for num_threads(width) shared(matrix)
	for(int j = 0; j < width; j++){

		
        int i = omp_get_thread_num();
        long long sum = 0;
        // create two vectors to send as reference to our functions
        std::vector<long double> ei;
        std::vector<long double> ej;
        
        findElements(i, j, kernel, width, std::ref(ei), std::ref(ej));
        
        // iterate over nodes that we need to calculate our current node
        for (int e = 0; e < ei.size(); e++){
            sum += matrix[ei[e]][ej[e]]*0.5+2;
        }
		#pragma omp barrier
        matrix[i][j] = sum;
        
    }

    auto finish = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::nanoseconds>(finish-start).count() / 1000000000.0<< "\n";
    std::cout<<matrix[9][9]<< std::endl;

}

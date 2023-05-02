#include <string>
#include <vector>
#include <ostream>
#include <iostream>
#include <fstream>

using namespace std;

template <typename T>
bool read_off(const std::string off_file,vector<T> &verticies,vector<T> &normals)
{
    std::ifstream input(off_file);
    if(!input) { return false; }
    std::string line;
    if(!std::getline(input,line)||line!="NOFF")
    {
        return false;
    }
    int nv,nf,ne;
    input >> nv >> nf >> ne;
    std::cout << nv << " " << nf << " " << ne << std::endl;
    T tmp;
    for(int v = 0; v<nv; ++v)
    {
        for(int i = 0; i<3; ++i)
        {
            input>>tmp;
            verticies.push_back(tmp);
        }
        for(int i = 0; i<3; ++i)
        {
            input>>tmp;
            normals.push_back(tmp);
        }
    }
    // we don't need faces
    return true;
}

template <typename T>
void export_result(const std::string output_file_name, const vector<T> &X, const vector<T> &FX)
{
    std::cout << "writing to file " << output_file_name << std::endl;
    std::ofstream output(output_file_name);
    output << "OFF" << std::endl;
    for (size_t i = 0; i < FX.size(); ++i)
    {
        output << X[i * 3 + 0] << " " << X[i * 3 + 1] << " " << X[i * 3 + 2] << " " << FX[i] << std::endl;
    }
}

int main(int argc,char *argv[])
{
    if(argc<2)
    {
        printf("usage: ./mls mesh.off [result.off]\n");
        exit(1);
    }
    std::cout << argc << std::endl;
    vector<double> vertices,normals;
    if(!read_off(argv[1],vertices,normals))
    {
        std::cout << "Error during read file " << argv[1] << std::endl;
        exit(1);
    }
   
    normals.resize(normals.size() / 3);
    if(argc == 3)
    {
        std::cout << "Writing to file " << argv[2] << std::endl;
        export_result(argv[2],vertices,normals);
    }
    return 0;
}

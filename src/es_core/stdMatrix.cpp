#include "es_core_pch.h"
#include "stdMatrix.h"

Eigen::MatrixXf Eigen::BuildMatFromFile(std::string fName)
{
    std::vector<std::vector<float>> tempMat;
    std::ifstream file(fName);
    std::string line;
    int count = 0;
    float temp;
    int row = 0;
    while (file.good()) {
        std::vector<float> row;
        std::getline(file, line);
        std::stringstream iss(line);
        std::string val;
        while (iss.good()) {
            std::getline(iss, val, ',');
            std::stringstream convertor(val);
            convertor >> temp;
            row.push_back(temp);
        }
        tempMat.push_back(row);
    }
    Eigen::MatrixXf outMat(tempMat.size(), tempMat[0].size());
    for (int i = 0; i < (int)tempMat.size(); i++) {
        for (int j = 0; j < (int)tempMat[i].size(); j++) {
            outMat(i, j) = tempMat[i][j];
        }
    }
    return outMat;
}

void Eigen::writeToCSVfile(std::string name, Eigen::MatrixXf matrix)
{
    std::ofstream file(name.c_str());
    file << matrix.format(CSVFormat);
}

void Eigen::removeColumn(Eigen::MatrixXf& matrix, unsigned int colToRemove)
{
    unsigned int numRows = (unsigned int)matrix.rows();
    unsigned int numCols = (unsigned int)matrix.cols() - 1;

    if (colToRemove < numCols)
        matrix.block(0, colToRemove, numRows, numCols - colToRemove) = matrix.rightCols(numCols - colToRemove);

    matrix.conservativeResize(numRows, numCols);
}

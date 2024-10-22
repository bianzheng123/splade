

#include <iostream>
#include <fstream>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <string.h>
#include <cstring>
#include <chrono>
#include <thread>
//#include <pthread.h>
#include "Preprocess.h"
#include "basis.h"
#include "alg.h"
#include "expe.h"
#include "rd_stdcout.h"


extern std::string data_fold, index_fold;
extern std::string data_fold1, data_fold2;
int mips2set::l = 10;
std::mt19937 mips2set::rng(int(std::time(0)));
std::uniform_real_distribution<float> mips2set::ur(0, 1);

std::string getNameByDate();

int main(int argc, char const *argv[]) {
    rdCoutStream rd(
            getNameByDate()); //Redirect the cout to a file and output it to the console at the same time (Create at Nov 13, 2023)

    std::string dataset = "base_small";
    std::string qname = "queries";

    int datasetID = -1;

    int L = 5;
    int K = 12;
    int k = 50;
    float c = 0.5;
    int l = 20;
    int ub = 10000;
    int m = 150;
    int b = 1;
    float F = 2.0f;
    int sin_m = 30, sin_h = 2;
    int mode = 0;

    if (argc > 1) mode = std::atoi(argv[1]);
    if (argc > 2) dataset = argv[2];
    if (argc > 3) qname = argv[3];

    //if (argc > 4) l = std::atoi(argv[3]);

    std::string argvStr[5];
    argvStr[1] = "datasets/" + dataset + ".csr";
    argvStr[2] = (dataset + ".index");
    argvStr[3] = ("datasets/" + dataset + "_all.ben");
    argvStr[4] = "datasets/" + qname + ".dev.csr";

    std::cout << "Using SOSIA for Sparse Dataset " << argvStr[1] << std::endl;
    Preprocess prep((argvStr[1]), (argvStr[4]), (argvStr[3]));
    std::vector<resOutput> res;
    switch (mode) {
        case 0:
            runAll(prep, c, k, m, ub, F, l, sin_m, sin_h, res);
            break;
        case 1:
            varyL(prep, c, k, m, ub, F, l, sin_m, sin_h, res);
            break;
        case 2:
            varyM(prep, c, k, m, ub, F, l, sin_m, sin_h, res);
            break;
        case 3:
            varyk(prep, c, k, m, ub, F, l, sin_m, sin_h, res);
            break;
        case 4:
            RecallTime(prep, c, k, m, ub, F, l, sin_m, sin_h, res);
            break;
    }

    saveAndShow(c, k, dataset, res);
    return 0;
}

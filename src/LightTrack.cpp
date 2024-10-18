//
// Created by xiongzhuang on 2021/10/8.
//
#include <utility>
#include <iostream>
#include <fstream>
#include <sstream>
#include <thread>
#include <chrono>
#include "time/timer.h"
#include "LightTrack/LightTrack.h"

std::vector<float> initTFData(6144);
std::vector<float> backboneTFData(31104);

float combineHalfFloats(cv::float16_t half1, cv::float16_t half2) {
    // 将半精度浮点数转换为 16 位整数表示
    uint16_t h1 = *(uint16_t*)&half1;
    uint16_t h2 = *(uint16_t*)&half2;

    uint32_t combined = (h2 << 16) | h1;
    return *(float*)&combined;
}

void sleep(int s)
{
    std::this_thread::sleep_for(std::chrono::seconds(s));
}

static void dump_tensor_attr(rknn_tensor_attr *attr)
{
    printf("  index=%d, name=%s, n_dims=%d, dims=[%d, %d, %d, %d], n_elems=%d, size=%d, fmt=%s, type=%s, qnt_type=%s, "
           "zp=%d, scale=%f\n \n",
           attr->index, attr->name, attr->n_dims, attr->dims[0], attr->dims[1], attr->dims[2], attr->dims[3],
           attr->n_elems, attr->size, get_format_string(attr->fmt), get_type_string(attr->type),
           get_qnt_type_string(attr->qnt_type), attr->zp, attr->scale);
}

void NCHWtoNHWC(const float* nchwData, float* nhwcData, int batchSize, int height, int width, int channels) {
    for (int b = 0; b < batchSize; ++b) {
        for (int h = 0; h < height; ++h) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < channels; ++c) {
                    // NCHW 到 NHWC 的索引映射
                    nhwcData[(b * height * width * channels) + (h * width * channels) + (w * channels) + c] =
                        nchwData[(b * channels * height * width) + (c * height * width) + (h * width) + w];
                }
            }
        }
    }
}

/*
    @addr https://github.com/Z-Xiong/LightTrack-rknn/issues/4
 */
// void Z_Xiong_Convert(void)
// {
//     float *p_zf_nchw = (float *)zf[0].buf;
//     std::vector<float> zf_nhwc(w*h*c);
//     float *p_zf_nhwc = zf_nhwc.data();
//     for (size_t i = 0; i < c; i++) {
//         for (size_t j = 0; j < h*w; j++) {
//             p_zf_nhwc[j * c + i] = p_zf_nchw[i*h*w + j];
//         }
//     }
// }

void print_img(cv::Mat &data, const char* file_name,int channel,int height,int width)
{
    printf("start to read input image data......\n");
    std::ofstream outFile(file_name);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            cv::Vec3b pixel = data.at<cv::Vec3b>(y, x);
            outFile << (int)pixel[2] << " " << (int)pixel[1] << " " << (int)pixel[0] << " \n"; //R G B 格式
        }
        outFile << std::endl;
    }
    outFile.close();
    printf("end to read input image data.\n");
}

void print_rknn(float *data, const char* file_name,int channel,int height,int width)
{
    std::ofstream OutFile(file_name);

    for (int i=0; i<channel*height*width; i++)
    {
        std::stringstream ss;
        std::string s;
        ss << data[i];
        s = ss.str();
        OutFile << s;
        OutFile << "\n";
    }
    OutFile.close();
}

inline float fast_exp(float x)
{
    union {
        uint32_t i;
        float f;
    } v{};
    v.i = (1 << 23) * (1.4426950409 * x + 126.93490512f);
    return v.f;
}

inline float sigmoid(float x)
{
    return 1.0f / (1.0f + fast_exp(-x));
}

static float sz_whFun(cv::Point2f wh)
{
    float pad = (wh.x + wh.y) * 0.5f;
    float sz2 = (wh.x + pad) * (wh.y + pad);
    return std::sqrt(sz2);
}

static std::vector<float> sz_change_fun(std::vector<float> w, std::vector<float> h,float sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    std::vector<float> pad(rows * cols, 0);
    std::vector<float> sz2;
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            pad[i*cols+j] = (w[i * cols + j] + h[i * cols + j]) * 0.5f;
        }
    }
    for (int i = 0; i < cols; i++)
    {
        for (int j = 0; j < rows; j++)
        {
            float t = std::sqrt((w[i * rows + j] + pad[i*rows+j]) * (h[i * rows + j] + pad[i*rows+j])) / sz;

            sz2.push_back(std::max(t,(float)1.0/t) );
        }
    }


    return sz2;
}

static std::vector<float> ratio_change_fun(std::vector<float> w, std::vector<float> h, const cv::Point2f& target_sz)
{
    int rows = int(std::sqrt(w.size()));
    int cols = int(std::sqrt(w.size()));
    float ratio = target_sz.x / target_sz.y;
    std::vector<float> sz2;
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            float t = ratio / (w[i * cols + j] / h[i * cols + j]);
            sz2.push_back(std::max(t, (float)1.0 / t));
        }
    }

    return sz2;
}

//LightTrack的解析函数，加载并初始化模型生成模型指针
LightTrack::LightTrack(const std::string& model_init, const std::string& model_backbone, const std::string& model_neck_head)
{

    score_size = int(round(this->instance_size / this->total_stride));

    int model_len=0;

    /********************加载模型***********************/
    std::cout << "---------START LOAD MODEL---------\n" <<std::endl;

    init_model_data = LightTrack::load_model(model_init.c_str(), &model_len);
    rknn_init(&(this->net_init), init_model_data, model_len, 0, NULL);
    printf("=============Load init model succeed. \n\n");

    backbone_model_data = LightTrack::load_model(model_backbone.c_str(), &model_len);
    rknn_init(&(this->net_backbone), backbone_model_data, model_len, 0, NULL);
    printf("=============Load backbone model succeed \n\n");

    neck_head_model_data = LightTrack::load_model(model_neck_head.c_str(), &model_len);
    rknn_init(&(this->net_neck_head), neck_head_model_data, model_len, 0, NULL);
    printf("=============Load neck_head model succeed \n\n");

    /**********************查询属性***************************/
    std::cout << "---------START QUERY MODEL ATTRBUTE---------\n" <<std::endl;

    rknn_query(this->net_init, RKNN_QUERY_IN_OUT_NUM, &init_io_num, sizeof(init_io_num));
    printf("net-init model input num: %d, output num: %d\n", init_io_num.n_input, init_io_num.n_output);
    memset(init_input_attrs, 0, sizeof(init_input_attrs));
    memset(init_output_attrs, 0, sizeof(init_output_attrs));
    printf("init model attrbute:\n");
    for (int i = 0; i < init_io_num.n_input; i++)
    {
        init_input_attrs[i].index = i;
        init_output_attrs[i].index = i;
        rknn_query(this->net_init, RKNN_QUERY_INPUT_ATTR, &(init_input_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(init_input_attrs[i]));
        rknn_query(this->net_init, RKNN_QUERY_OUTPUT_ATTR, &(init_output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(init_output_attrs[i]));
    }

    rknn_query(this->net_backbone, RKNN_QUERY_IN_OUT_NUM, &backbone_io_num, sizeof(backbone_io_num));
    printf("backbone model input num: %d, output num: %d\n", backbone_io_num.n_input, backbone_io_num.n_output);
    memset(backbone_input_attrs, 0, sizeof(backbone_input_attrs));
    memset(backbone_output_attrs, 0, sizeof(backbone_output_attrs));
    for (int i = 0; i < backbone_io_num.n_input; i++)
    {
        backbone_input_attrs[i].index = i;
        backbone_output_attrs[i].index = i;
        rknn_query(this->net_backbone, RKNN_QUERY_INPUT_ATTR, &(backbone_input_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(backbone_input_attrs[i]));
        rknn_query(this->net_backbone, RKNN_QUERY_OUTPUT_ATTR, &(backbone_output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(backbone_output_attrs[i]));
    }

    rknn_query(this->net_neck_head, RKNN_QUERY_IN_OUT_NUM, &neck_head_io_num, sizeof(neck_head_io_num));
    printf("neck_head model input num: %d, output num: %d\n", neck_head_io_num.n_input, neck_head_io_num.n_output);
    memset(neck_head_input_attrs, 0, sizeof(neck_head_input_attrs));
    memset(neck_head_output_attrs, 0, sizeof(neck_head_output_attrs));
    for (int i = 0; i < neck_head_io_num.n_input; i++)
    {
        neck_head_input_attrs[i].index = i;
        neck_head_output_attrs[i].index = i;
        rknn_query(this->net_neck_head, RKNN_QUERY_INPUT_ATTR, &(neck_head_input_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(neck_head_input_attrs[i]));
        rknn_query(this->net_neck_head, RKNN_QUERY_OUTPUT_ATTR, &(neck_head_output_attrs[i]), sizeof(rknn_tensor_attr));
        dump_tensor_attr(&(neck_head_output_attrs[i]));
    }

    /*****************************查询版本信息*******************************/
    std::cout << "---------START QUERY SDK/Driver VERSION---------\n" << std::endl;
    rknn_sdk_version version;
    rknn_query(this->net_init, RKNN_QUERY_SDK_VERSION, &version, sizeof(rknn_sdk_version));
    printf("sdk api version: %s\n", version.api_version);
    printf("driver version: %s\n", version.drv_version);
}

LightTrack::~LightTrack()
{
    rknn_outputs_release(net_init, 1, zf);

    rknn_destroy(net_init);
    rknn_destroy(net_backbone);
    rknn_destroy(net_neck_head);

    if (init_model_data)
    {
        free(init_model_data);
    }

    if (backbone_model_data)
    {
        free(backbone_model_data);
    }

    if (neck_head_model_data)
    {
        free(neck_head_model_data);
    }
}

// 初始化LightTrack追踪器
void LightTrack::init(const cv::Mat& img, cv::Point target_pos_, cv::Point2f target_sz_)
{
    this->target_pos = std::move(target_pos_);
    this->target_sz = std::move(target_sz_);

    std::cout << "init target pos: " << target_pos << std::endl;
    std::cout << "init target_sz: " << target_sz << std::endl;

    this->grids();  //创建模型特征图（二维矩阵）

    // 对模板图像而言：在第一帧以s_z为边长，以目标中心为中心点，截取图像补丁（如果超出第一帧的尺寸，用均值填充）。之后将其resize为127x127x3.成为模板图像
    // context = 1/2 * (w+h) = 2*pad
    float wc_z = target_sz.x + this->context_amount * (target_sz.x + target_sz.y);
    float hc_z = target_sz.y + this->context_amount * (target_sz.x + target_sz.y);
    // z_crop size = sqrt((w+2p)*(h+2p))
    float s_z = round(sqrt(wc_z * hc_z));   // orignal size 均值填充模板图像尺寸（正方形边长）相当一个算子框

    cv::Scalar avg_chans = cv::mean(img);  //返回一帧图像RGB各通道均值【，，】
    cv::Mat z_crop;

    z_crop  = get_subwindow_tracking(img, target_pos, this->exemplar_size, int(s_z));//将roi区域resize成net-neck模型输入

    // Set Input Data
    cv::Mat rgb;
    cv::cvtColor(z_crop, rgb, cv::COLOR_BGR2RGB);
    // cv::imwrite("input.jpg",rgb);

    //先紧靠一张图片验证fp16对齐(ONNX:input{NCHW}  output{NHWC})
    sleep(2);
    cv::Mat Input_Image = cv::imread("input.jpg");  //bgr
    print_img(Input_Image,"Input_Image_Data.txt",3,127,127);
    cv::Mat Input_Image_RGB;
    // cv::cvtColor(Input_Image, Input_Image_RGB, cv::COLOR_BGR2RGB);
    std::cout << "Input Image size [cols , rows , channels] : " << "[" << Input_Image_RGB.cols << "," << Input_Image_RGB.rows << "," << Input_Image_RGB.channels() << "]" << std::endl;

    rknn_input rknn_img[1];
    memset(rknn_img, 0, sizeof(rknn_img));
    rknn_img[0].index = 0;
    rknn_img[0].type = RKNN_TENSOR_UINT8;                 //设置成这主要是8UC3
    rknn_img[0].size = Input_Image.cols*rgb.rows*rgb.channels();  //[w*h*c]总像素数（因为type不用×2）
    rknn_img[0].fmt = RKNN_TENSOR_NHWC;
    rknn_img[0].buf = Input_Image.data;
    rknn_inputs_set(net_init, 1, rknn_img);

    // Run
    rknn_run(net_init, nullptr);

    // Get Output
    memset(zf, 0, sizeof(zf));
    for (auto & i : zf) {
        i.want_float = 1;  //要将输出数据转为 float 类型输出(1)，但我却是以set_input设置输出，如果按照版主设置内存站四字节
        i.is_prealloc = 0; //不是预分配
    }
    rknn_outputs_get(net_init, 1, zf, nullptr);

    //重新排序
    std::cout << "model size: " << zf[0].size << std::endl;
    float *zf_data = (float *)zf[0].buf;
    print_rknn(zf_data,"01_init_model_output.txt",96,8,8);
    initTFData.clear();
    NCHWtoNHWC(zf_data,initTFData.data(),1,8,8,96);
    // print_rknn(initTFData.data(),"02_init_model_convert_output.txt",96,8,8);

    std::vector<float> hanning(this->score_size,0);  // 18

    this->window.resize(this->score_size * this->score_size, 0);
    for (int i = 0; i < this->score_size; i++)
    {
        float w = 0.5f - 0.5f * std::cos(2 * PI * float(i) / float(this->score_size - 1));
        hanning[i] = w;         //保存net——neck输出结果
    }
    for (int i = 0; i < this->score_size; i++)
    {

        for (int j = 0; j < this->score_size; j++)
        {
            this->window[i*this->score_size+j] = hanning[i] * hanning[j];  //将图像分割
        }
    }
}

void LightTrack::update(const cv::Mat &x_crop, float scale_z)
{
    time_checker time2{}, time3{}, time4{}, time5{};

    /* net backbone */
    time2.start();
    // Set Input Data
    rknn_input rknn_img[1];
    memset(rknn_img, 0, sizeof(rknn_img));
    cv::cvtColor(x_crop, x_crop, cv::COLOR_BGR2RGB);
    rknn_img[0].index = 0;
    rknn_img[0].type = RKNN_TENSOR_UINT8;
    rknn_img[0].size = x_crop.cols*x_crop.rows*x_crop.channels();
    rknn_img[0].fmt = RKNN_TENSOR_NHWC;
    rknn_img[0].buf = x_crop.data;
    rknn_inputs_set(net_backbone, 1, rknn_img);
    time2.stop();
    time2.show_distance("Update stage ---- input seting cost time");

    time3.start();
    // Run
    rknn_run(net_backbone, nullptr);
    // Get Output
    rknn_output xf[1];
    memset(xf, 0, sizeof(xf));
    for (auto & i : xf) {
        i.want_float = 1;
        i.is_prealloc = 0;
    }
    rknn_outputs_get(net_backbone, 1, xf, nullptr);

    std::cout << "model size: " << xf[0].size << std::endl;
    float *xf_data = (float *)xf[0].buf;
    // print_rknn(xf_data,"01_backbone_model_output.txt",96,18,18);
    backboneTFData.clear();
    NCHWtoNHWC(xf_data,backboneTFData.data(),1,18,18,96);
    // print_rknn(backboneTFData.data(),"02_backbone_model_convert_output.txt",96,18,18);

    time3.stop();
    time3.show_distance("Update stage ---- output xf extracting cost time");

    /* net neck head */
    time4.start();
    rknn_input zf_xf[2];
    memset(zf_xf, 0, sizeof(zf_xf));
    zf_xf[0].index = 0;
    zf_xf[0].type = RKNN_TENSOR_FLOAT32;
    zf_xf[0].size = zf[0].size;
    zf_xf[0].fmt = RKNN_TENSOR_NHWC;
    zf_xf[0].buf = initTFData.data();
    zf_xf[0].pass_through = 0;   // 这里必须为0，当为1时设置的fmt将不起作用，就会时默认的NHWC，注意rknn的输入都是NHWC格式，输出都是NCHW格式，所以这里的xf和zf都是NCHW格式
    zf_xf[1].index = 1;
    zf_xf[1].type = RKNN_TENSOR_FLOAT32;
    zf_xf[1].size = xf[0].size;
    zf_xf[1].fmt = RKNN_TENSOR_NHWC;
    zf_xf[1].buf = backboneTFData.data();
    zf_xf[1].pass_through = 0;
    int ret = rknn_inputs_set(net_neck_head, 2, zf_xf);

    /************设置属性****************/

    rknn_run(net_neck_head, nullptr);
    rknn_output outputs[2];
    memset(outputs, 0, sizeof(outputs));
    for (auto & output : outputs) {
        output.want_float = 1;
        output.is_prealloc = 0;
    }
    rknn_outputs_get(net_neck_head, 2, outputs, nullptr);

    float* nf1_data = (float*)outputs[0].buf;
    // print_rknn(nf1_data,"neck_model_output_01.txt",1,18,18);

    float* nf2_data = (float*)outputs[1].buf;
    // print_rknn(nf2_data,"neck_model_output_02.txt",4,18,18);

    time4.stop();
    time4.show_distance("Update stage ---- output cls_score and bbox_pred extracting cost time");

    time5.start();
    // manually call sigmoid on the output
    std::vector<float> cls_score_sigmoid;

    float *cls_score_data = (float *)outputs[0].buf;
    float *bbox_score_data = (float *)outputs[1].buf;
    cls_score_sigmoid.clear();

    int cols = score_size;  //18是最后一个模型的长度和宽度
    int rows = score_size;

    for (int i = 0; i < cols*rows; i++)   // 18 * 18
    {
        cls_score_sigmoid.push_back(sigmoid(cls_score_data[i]));
    }

    std::vector<float> pred_x1(cols*rows, 0), pred_y1(cols*rows, 0), pred_x2(cols*rows, 0), pred_y2(cols*rows, 0);

    float* bbox_pred_data1 = (float*)outputs[1].buf;
    float* bbox_pred_data2 = (float*)outputs[1].buf + cols*rows;
    float* bbox_pred_data3 = (float*)outputs[1].buf + 2*cols*rows;
    float* bbox_pred_data4 = (float*)outputs[1].buf + 3*cols*rows;
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            pred_x1[i*cols + j] = this->grid_to_search_x[i*cols + j] - bbox_pred_data1[i*cols + j];
            pred_y1[i*cols + j] = this->grid_to_search_y[i*cols + j] - bbox_pred_data2[i*cols + j];
            pred_x2[i*cols + j] = this->grid_to_search_x[i*cols + j] + bbox_pred_data3[i*cols + j];
            pred_y2[i*cols + j] = this->grid_to_search_y[i*cols + j] + bbox_pred_data4[i*cols + j];
        }
    }

    // size penalty (1)
    std::vector<float> w(cols*rows, 0), h(cols*rows, 0);
    for (int i=0; i<rows; i++)
    {
        for (int j=0; j<cols; j++)
        {
            w[i*cols + j] = pred_x2[i*cols + j] - pred_x1[i*cols + j];
            h[i*rows + j] = pred_y2[i*rows + j] - pred_y1[i*cols + j];
        }
    }

    float sz_wh = sz_whFun(target_sz);
    std::vector<float> s_c = sz_change_fun(w, h, sz_wh);
    std::vector<float> r_c = ratio_change_fun(w, h, target_sz);

    std::vector<float> penalty(rows*cols,0);
    for (int i = 0; i < rows * cols; i++)
    {
        penalty[i] = std::exp(-1 * (s_c[i] * r_c[i]-1) * this->penalty_tk);
    }

    // window penalty
    std::vector<float> pscore(rows*cols,0);
    int r_max = 0, c_max = 0;
    float maxScore = 0;
    for (int i = 0; i < rows * cols; i++)
    {
        pscore[i] = (penalty[i] * cls_score_sigmoid[i]) * (1 - this->window_influence) + this->window[i] * this->window_influence;
        if (pscore[i] > maxScore)
        {
            // get max
            maxScore = pscore[i];
            r_max = std::floor(i / rows);
            c_max = ((float)i / rows - r_max) * rows;
        }
    }

    time5.stop();
    time5.show_distance("Update stage ---- postprocess cost time");
    std::cout << "pscore_window max score is: " << pscore[r_max * cols + c_max] << std::endl;

    // to real size
    float pred_x1_real = pred_x1[r_max * cols + c_max]; // pred_x1[r_max, c_max]
    float pred_y1_real = pred_y1[r_max * cols + c_max];
    float pred_x2_real = pred_x2[r_max * cols + c_max];
    float pred_y2_real = pred_y2[r_max * cols + c_max];

    float pred_xs = (pred_x1_real + pred_x2_real) / 2;
    float pred_ys = (pred_y1_real + pred_y2_real) / 2;
    float pred_w = pred_x2_real - pred_x1_real;
    float pred_h = pred_y2_real - pred_y1_real;

    float diff_xs = pred_xs - float(this->instance_size) / 2;
    float diff_ys = pred_ys - float(this->instance_size) / 2;

    diff_xs /= scale_z;
    diff_ys /= scale_z;
    pred_w /=scale_z;
    pred_h /= scale_z;

    target_sz.x = target_sz.x / scale_z;
    target_sz.y = target_sz.y / scale_z;

    // size learning rate
    float lr_new = penalty[r_max * cols + c_max] * cls_score_sigmoid[r_max * cols + c_max] * this->lr;

    // size rate
    float res_xs = float (target_pos.x) + diff_xs;
    float res_ys = float (target_pos.y) + diff_ys;
    float res_w = pred_w * lr_new + (1 - lr_new) * target_sz.x;
    float res_h = pred_h * lr_new + (1 - lr_new) * target_sz.y;

    target_pos.x = int(res_xs);
    target_pos.y = int(res_ys);

    target_sz.x = target_sz.x * (1 - lr_new) + lr_new * res_w;
    target_sz.y = target_sz.y * (1 - lr_new) + lr_new * res_h;

    rknn_outputs_release(net_neck_head, 2, outputs);
    rknn_outputs_release(net_backbone, 1, xf);
}

void LightTrack::track(const cv::Mat& im)
{
    time_checker time1{};
    //上一帧数据
    float hc_z = target_sz.y + this->context_amount * (target_sz.x + target_sz.y);  //基本和track_init一样
    float wc_z = target_sz.x + this->context_amount * (target_sz.x + target_sz.y);
    float s_z = sqrt(wc_z * hc_z);  // resize roi size
    float scale_z = float(this->exemplar_size) / s_z;
    std::cout << "-------" << scale_z << "-------" << std::endl;

    float d_search = float(this->instance_size - this->exemplar_size) / 2;  //输入模型的尺度 backbone_model_size - init_model_size = 288-127
    float pad = d_search / scale_z;
    float s_x = s_z + 2 * pad;


    time1.start();
    cv::Mat x_crop;
    //构建追踪子窗口
    x_crop  = get_subwindow_tracking(im, target_pos, this->instance_size, int(s_x));
    time1.stop();
    time1.show_distance("Update stage ---- get subwindow cost time");

    // update
    target_sz.x = target_sz.x * scale_z;
    target_sz.y = target_sz.y * scale_z;

    this->update(x_crop, scale_z);
    target_pos.x = std::max(0, min(im.cols, target_pos.x));
    target_pos.y = std::max(0, min(im.rows, target_pos.y));
    target_sz.x = float(std::max(10, min(im.cols, int(target_sz.x))));
    target_sz.y = float(std::max(10, min(im.rows, int(target_sz.y))));

    std::cout << "track target pos: " << target_pos << std::endl;
    std::cout << "track target_sz: " << target_sz << std::endl;
}

//解析输入模型，将模型二进制数据返回，并将模型大小返回到model_size
unsigned char *LightTrack::load_model(const char *filename, int *model_size)
{
    FILE *fp = fopen(filename, "rb");
    if(fp == nullptr) {
        printf("fopen %s fail!\n", filename);
        return nullptr;
    }
    printf("Load model %s succeed!\n", filename);
    fseek(fp, 0, SEEK_END);
    int model_len = ftell(fp);
    auto *model = (unsigned char*)malloc(model_len);
    fseek(fp, 0, SEEK_SET);
    if(model_len != fread(model, 1, model_len, fp)) {
        printf("fread %s fail!\n", filename);
        free(model);
        return NULL;
    }
    *model_size = model_len;
    fclose(fp);
    return model;
}

void LightTrack::grids()
{
    /*
    输入搜索图像每个特征元素
    :return: H*W*2 (每个元素位置)
    */
    int sz = score_size;   // 18

    //创建一个18*18大小的容器初始值为0（数量：324）
    this->grid_to_search_x.resize(sz * sz, 0);
    this->grid_to_search_y.resize(sz * sz, 0);

    for (int i = 0; i < sz; i++)
    {
        for (int j = 0; j < sz; j++)
        {
            this->grid_to_search_x[i*sz+j] = j*total_stride;   // 0~18*16 = 0~288
            this->grid_to_search_y[i*sz+j] = i*total_stride;
        }
    }
}

/*
    @function get_subwindow_tracking 子窗口追踪框
    @param im 输入图像
    @param pos 输入图像要追踪的目标中点
    @param model_sz 自定义算子大小
    @param original_sz 感觉是背景框
*/
cv::Mat LightTrack::get_subwindow_tracking(const cv::Mat& im, cv::Point2f pos, int model_sz, int original_sz)
{
    time_checker time1,time2, time3;
    time1.start();
    float c = (float)(original_sz + 1) / 2;  //中心点（初始尺度框中点，并非标定追踪框）
    printf("###########target_x:%0.2f  ,target_y:%0.2f  ,len:%0.2f .\n",pos.x,pos.y,c);
    //新框的四个坐标点
    int context_xmin = std::round(pos.x - c);
    int context_xmax = context_xmin + original_sz - 1;
    int context_ymin = std::round(pos.y - c);
    int context_ymax = context_ymin + original_sz - 1;
    //防止出界
    int left_pad = int(std::max(0, -context_xmin));
    int top_pad = int(std::max(0, -context_ymin));
    int right_pad = int(std::max(0, context_xmax - im.cols + 1));
    int bottom_pad = int(std::max(0, context_ymax - im.rows + 1));

    //可以修复，防止出界（如果出界回复成边界点）
    context_xmin += left_pad;
    context_xmax += left_pad;
    context_ymin += top_pad;
    context_ymax += top_pad;
    cv::Mat im_path_original;
    time1.stop();
    time1.show_distance("get_subwindow_tracking cost time 1:");

    if (top_pad > 0 || left_pad > 0 || right_pad > 0 || bottom_pad > 0)
    {
        time2.start();
        //创建一个巨大画布（>=原图像）
        cv::Mat te_im = cv::Mat::zeros(im.rows + top_pad + bottom_pad, im.cols + left_pad + right_pad, CV_8UC3);
        //te_im(cv::Rect(left_pad, top_pad, im.cols, im.rows)) = im;
        printf("#calibration box x_min: %d  #calibration box y_min:%d  #calibration box x_max:%d  #calibration box y_max:%d\n",context_xmin,context_ymin,context_xmax,context_ymax);
        // cv::copyMakeBorder(im, te_im, context_xmin, context_ymin, context_xmax, context_ymax, cv::BORDER_CONSTANT, 0.f);
        cv::copyMakeBorder(im, te_im, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT, 0.f);//画框
        im_path_original = te_im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));//ROI区域
        time2.stop();
        time2.show_distance("get_subwindow_tracking cost time 2-1-1:");
    }
    else
        im_path_original = im(cv::Rect(context_xmin, context_ymin, context_xmax - context_xmin + 1, context_ymax - context_ymin + 1));

    time3.start();
    cv::Mat im_path;
    cv::resize(im_path_original, im_path, cv::Size(model_sz, model_sz));
    time3.stop();
    time3.show_distance("get_subwindow_tracking cost time 2-1-2:");

    return im_path;
}

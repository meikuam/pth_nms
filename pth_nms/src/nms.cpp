#include <torch/extension.h>
#include <math.h>

#define CHECK_CPU(x) AT_ASSERTM(!x.type().is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_DIMS(x) AT_ASSERTM(x.dim() == 4, #x " must have 4 dimensions")

#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Float, #x " must be float Tensor")
#define CHECK_INT(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Int, #x " must be int Tensor")
#define CHECK_LONG(x) AT_ASSERTM(x.type().scalarType() == torch::ScalarType::Long, #x " must be long Tensor")

using namespace at;
namespace torch {
int cpu_nms(
    torch::Tensor keep_out,
    torch::Tensor num_out,
    torch::Tensor boxes,
    torch::Tensor order,
    torch::Tensor areas,
    float nms_overlap_thresh
) {
    // boxes has to be sorted
    CHECK_INPUT(keep_out);  CHECK_LONG(keep_out);
    CHECK_INPUT(num_out);   CHECK_LONG(num_out);
    CHECK_INPUT(boxes);     CHECK_FLOAT(boxes);
    CHECK_INPUT(order);     CHECK_LONG(order);
    CHECK_INPUT(areas);     CHECK_FLOAT(areas);

    // Number of ROIs
    long boxes_num = boxes.size(0);
    long boxes_dim = boxes.size(1);

    long * keep_out_flat = keep_out.data<long>();
    float * boxes_flat = boxes.data<float>();
    long * order_flat = order.data<long>();
    float * areas_flat = areas.data<float>();

    torch::Tensor suppressed = torch::zeros({boxes_num}, torch::kInt);
    int * suppressed_flat =  suppressed.data<int>(); 

    // nominal indices
    int i, j;
    // sorted indices
    int _i, _j;
    // temp variables for box i's (the box currently under consideration)
    float ix1, iy1, ix2, iy2, iarea;
    // variables for computing overlap with box j (lower scoring box)
    float xx1, yy1, xx2, yy2;
    float w, h;
    float inter, ovr;

    long num_to_keep = 0;
    for (_i=0; _i < boxes_num; ++_i) {
        i = order_flat[_i];
        if (suppressed_flat[i] == 1) {
            continue;
        }
        keep_out_flat[num_to_keep++] = i;
        ix1 = boxes_flat[i * boxes_dim];
        iy1 = boxes_flat[i * boxes_dim + 1];
        ix2 = boxes_flat[i * boxes_dim + 2];
        iy2 = boxes_flat[i * boxes_dim + 3];
        iarea = areas_flat[i];
        for (_j = _i + 1; _j < boxes_num; ++_j) {
            j = order_flat[_j];
            if (suppressed_flat[j] == 1) {
                continue;
            }
            xx1 = fmaxf(ix1, boxes_flat[j * boxes_dim]);
            yy1 = fmaxf(iy1, boxes_flat[j * boxes_dim + 1]);
            xx2 = fminf(ix2, boxes_flat[j * boxes_dim + 2]);
            yy2 = fminf(iy2, boxes_flat[j * boxes_dim + 3]);
            w = fmaxf(0.0, xx2 - xx1 + 1);
            h = fmaxf(0.0, yy2 - yy1 + 1);
            inter = w * h;
            ovr = inter / (iarea + areas_flat[j] - inter);
            if (ovr >= nms_overlap_thresh) {
                suppressed_flat[j] = 1;
            }
        }
    }

    long *num_out_flat = num_out.data<long>();
    *num_out_flat = num_to_keep;
    return 1;
}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def(
      "nms",
      &torch::cpu_nms,
      "non maximum suppression CPU");
}

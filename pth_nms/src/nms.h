namespace torch {
int cpu_nms(
    torch::Tensor keep_out, // long
    torch::Tensor num_out, // long
    torch::Tensor boxes, // float
    torch::Tensor order, //long
    torch::Tensor areas, //float
    float nms_overlap_thresh
);
}

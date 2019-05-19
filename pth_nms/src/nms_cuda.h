namespace torch {
int gpu_nms(
    torch::Tensor keep_out, // long
    torch::Tensor num_out, // long
    torch::Tensor boxes, // cuda tensor
    float nms_overlap_thresh
);
}

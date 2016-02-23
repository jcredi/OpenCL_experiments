kernel void IncrementNumber(global float4 *celldata_in, global float4 *celldata_out) {
    int index = get_global_id(0);

    float4 a = celldata_in[index];
    a.w = a.w + 1;

    celldata_out[index] = a;  
}
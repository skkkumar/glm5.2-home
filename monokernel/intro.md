Mono kernel

What is the issue at low concurrency / low context?

Launch bound - launch/drain time more than time to transfer weights/inputs and do the computation




Roofline plot



What to do if you are launch bound ? Use waves to hide latency

SIMD have the capability to add waves/resident threads - program state
Use waves to hide the launch time 
We use 2 waves in monokernel - when one wave is doing the computation, the other wave can be loading the data
Need to remember resident waves not computing share registers. So we need to tradeoff by selecting minimum occupancy. Refer - https://rocm.blogs.amd.com/software-tools-optimization/occupancy-math-mi355x/README.html 

Time —->

Wave 1
compute


Wave 2
Load data
compute



Why are we limited to 12 tokens?

There are physical limitations of LDS, registers, waves.
The SIMD has 8 waves maximum.
The Acc VGPR supports 16 rows so we should be able to do 16 concurrency
Each CU has 160kB LDS

In monokernel, we use the LDS to load the inputs and store the processed inputs for the next layer there. The weights are loaded from HBM directly.

In GLM5.2 case, we hit the LDS limit by 12 tokens for q_b layer (largest input size of the model)
input size = rows x 6144, bf16 data type

8 rows
96 KB
12 rows 
144 kb
16 rows
192 kb 



How does monokernel use waves design? Each wave and CU does the same layer work for simplicity

Each layer is executed one after each other
Each layer is written in steps of 256 CUs work called stage
Each CU work is called task, so 256 tasks
We need to write the 8 waves per CUs or CTA level
Thread count = 2 waves per SIMD x 4 SIMDs per CUs x 64 threads for the 64 lanes = 512 threads


u_layer
Load the weights from the HBM
Hint_wait, poll
Wait for mail box to be available
stage_x
Copy input from mailbox rows to LDS
Mma_units, etc. 
Run the gemms, point operations, etc.
reduce_rows
Combine results for gemm
Emit_out, put_bf
Tile goes into LDS, tag the next mailbox


 How does a monokernel differ from normal code?

Gif animations? Ok 

What is a mailbox? How does it help to schedule LDS read/write for the threads?



Sample cases
How to write gemm type of operations?

How to write indexer layer?


How to write tp all reduce?



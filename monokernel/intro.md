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



# How does monokernel use waves design? Each wave and CU does the same layer work for simplicity

Each layer is executed one after each other. Each layer is written in steps of 256 CUs work called stage.

 for t in range(start("q_b"), N_QB, G):  : # G = 256 
            stamp("q_b", t, 0)

qkv_a
q_b
index_score
…


Each CU work is called task, so 256 tasks

| task 1 | Task 2 | Task 3 |   |   |   |   |          |
|--------|--------|--------|---|---|---|---|----------|
|        |        |        |   |   |   |   |          |
|        |        |        |   |   |   |   |          |
|        |        |        |   |   |   |   |          |
|        |        |        |   |   |   |   |          |
|        |        |        |   |   |   |   |          |
|        |        |        |   |   |   |   |          |
|        |        |        |   |   |   |   | task 256 |


We need to write the wave level code
Thread count = 2 waves per SIMD x 4 SIMDs per CUs x 64 threads for the 64 lanes = 512 threads

@flyc.kernel(known_block_size=[THREADS, 1, 1])
    def glm5_monokernel(
         tid = fx.thread_idx.x
        bid = fx.block_idx.x
        lane = tid % 64
        wave = tid // 64

| u_layer                     | Load the weights from the HBM            | def u_qb(c):                                   |
|-----------------------------|------------------------------------------|------------------------------------------------|
| Hint_wait, poll             | Wait for mail box to be available        | hint_wait(S,  lambda s: (mb("q_an")            |
| stage_x                     | Copy input from mailbox rows to LDS      | stage_x_pairs("q_an"                           |
| Mma_units, stage_x_rmsnorm  | Run the gemms, point operations, etc.    | acc = run_units(u_qb, QB_UNITS, QB_UNITS, pre) |
| reduce_rows                 | Combine results for gemm                 | reduce_rows(1, acc, emit_out(Q_B_TILE))        |
| Emit_out, put_bf            | Tile goes into LDS, tag the next mailbox | put_bf( mb("q_nope"),                          |






 How does a monokernel differ from normal code?

Gif animations? Ok 

What is a mailbox? How does it help to schedule LDS read/write for the threads?

Mailbox in the L2 scratch, an extension of HBM memory
Stores as (value, tag) 
Value is the value of input tensor, then loaded into LDS
Tag is step × launches_per_step + layer + 1
Each thread checks if previous thread finished and updates the tag

|      Launch      |  What q_norm writes into q_an  | Tag |
|:----------------:|:------------------------------:|:---:|
| step 0, layer 0  | layer 0's values               | 1   |
| step 0, layer 1  | layer 1's values, over the top | 2   |
| …                | …                              | …   |
| step 0, layer 77 | layer 77's values              | 78  |
| step 1, layer 0  | layer 0's values again         | 79  |



Sample cases
How to write gemm type of operations?

W_q_b = [4096, 2048]
Each task picks 16 rows, all columns
4096 rows / 16 rows = 256 CUs
For each task, the 8 waves per CU split the 2048 into 8 parts - 256 columns each. 
Then we use reduce rows to all across columns
Not all operations split 8 ways and use reduce rows

|         | 256 col |        |        |      |   |   |   |   |   |
|---------|---------|--------|--------|------|---|---|---|---|---|
| 16 rows | Wave 0  | Wave 1 | Wave 2 | wave |   |   |   |   | 8 |

|  Stage | Weight (per GPU, TP4) |        Rows per task       | Waves per row group | K-slices per wave |    Reduce?   |
|:------:|:---------------------:|:--------------------------:|:-------------------:|:-----------------:|:------------:|
| qkv_a  | 2624 × 6144           | 16                         | 8 on one group      | 1/8 of K          | yes, 8 waves |
| q_b    | 4096 × 2048           | 16                         | 8 on one group      | 1/8 of K          | yes, 8 waves |
| uk     | 16 heads × 192 × 512  | 128 (8 groups of 16)       | 1 wave per group    | all of K (192)    | no           |
| uv     | 16 heads × 256 × 512  | 64 (4 groups)              | 2 waves per group   | 1/2 of K          | yes, 2 waves |
| o      | 6144 × 4096           | 32 (2 groups)              | 4 waves per group   | 1/4 of K          | yes, 4 waves |
| router | 256 × 6144            | 8 (half a group, K-folded) | 8 on one group      | 1/8 of K, folded  | yes          |
| down   | 6144 × 512 per expert | dn_tile(S) dependent       | varies              | varies            | yes          |

If rows / 256 < 16 dont split it
E.g. uk rows=16
O, rows = 6144 / 16 / 256 = 1.5, so instead we don’t split it
Uv, rows = 512, too short
Waves can be grouped 
E.g. 2 groups so 1st group (0-3 waves) get 0-15 and 2nd group (4-7 waves) get rows 16-31 rows
O - 6144 / 32 = 192 tasks < 256 CUs/ each CU gets  1024 columns


|         | 1024 columns | 1024   |        |        |
|---------|--------------|--------|--------|--------|
| 16 rows | Wave 0       | Wave 1 | Wave 2 | Wave 3 |
| 16 rows | Wave 4       | 5      | 6      | 7      |

point wise operations 

Stage_x_rmsnorm does rms norm
It calls ss to do partial sum
Reduce_sums similar to reduce rows


How to write indexer layer?


How to write tp all reduce?



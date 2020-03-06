extern crate mpi;

use mpi::request::WaitGuard;
use mpi::traits::*;
use mpi::collective::{self, SystemOperation, UnsafeUserOperation};

fn main() {
  let universe = mpi::initialize().unwrap();
  let world = universe.world();
  let size = world.size();
  let rank : i32 = world.rank();
  let root_rank = 0;

  let next_rank = if rank + 1 < size { rank + 1 } else { 0 };
  let previous_rank = if rank > 0 { rank - 1 } else { size - 1 };

  let n = 1000;
  let data = vec![rank; n];
  // let data = vec![rank, rank * 2, rank * 3];

  mpi::request::scope(|scope| {
    println!("Sending from rank {} to {}", rank, next_rank);

    // Try Isend and Irecv
    let _sreq = WaitGuard::from(
      world
        .process_at_rank(next_rank)
        .immediate_send(scope, &data[..]),
    );

    let (result, _status) : (Vec<i32>, mpi::point_to_point::Status) = world.any_process().receive_vec();

    println!("data is previously {}, now {}", data[0], result[0]);

    // Try reduce to root
    let mut sum : i32 = 0;
    if rank == root_rank {
      world
        .this_process()
        .reduce_into_root(&rank, &mut sum, SystemOperation::sum());

      println!("Reduce result is {} on root", sum);
    } else {
      world
        .process_at_rank(root_rank)
        .reduce_into(&rank, SystemOperation::sum());
    }

    // Try all_reduce
    let mut sum_all : i32 = 0;
    world.all_reduce_into(&rank, &mut sum_all, SystemOperation::sum());
    println!("All reduce result is {}", sum_all);
  });
}

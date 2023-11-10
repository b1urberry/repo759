#include <iostream>
#include <vector>
#include <mpi.h>

int main(int argc, char *argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " n" << std::endl;
        return 1;
    }

    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int n = std::atoi(argv[1]);
    std::vector<float> buffer(n, 0.0f); // Initialize buffer with zeros

    // Fill the buffer array with float-type numbers
    for (int i = 0; i < n; ++i) {
        buffer[i] = static_cast<float>(i);
    }

    double t0 = 0.0, t1 = 0.0, total_time;

    if (rank == 0) {
        // Process 0 sends and then receives the message
        t0 = MPI_Wtime();
        MPI_Send(buffer.data(), n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(buffer.data(), n, MPI_FLOAT, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        t0 = (MPI_Wtime() - t0) * 1000; // Convert to milliseconds
    } else if (rank == 1) {
        // Process 1 receives and then sends the message
        t1 = MPI_Wtime();
        MPI_Recv(buffer.data(), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Send(buffer.data(), n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
        t1 = (MPI_Wtime() - t1) * 1000; // Convert to milliseconds
    }

    // Sum the times from both processes
    MPI_Reduce(&t0, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&t1, &total_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Print the total time in process 0
    if (rank == 0) {
        std::cout << total_time  << std::endl;
    }

    MPI_Finalize();
    return 0;
}

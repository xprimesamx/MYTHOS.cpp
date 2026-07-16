FROM ubuntu:22.04 AS builder

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    cmake \
    g++-12 \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .

RUN cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DOIL_BUILD_TESTS=OFF \
    -DOIL_BUILD_BENCHMARKS=OFF \
    -DCMAKE_CXX_COMPILER=g++-12 \
    && cmake --build build --parallel

FROM ubuntu:22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/build/tools/oil_infer .
COPY --from=builder /app/build/tools/oil_convert .
COPY --from=builder /app/build/tools/oil_bench .

EXPOSE 8080

ENTRYPOINT ["./oil_infer"]

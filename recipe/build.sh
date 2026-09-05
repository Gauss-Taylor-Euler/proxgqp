set -euxo pipefail

cmake -S "${SRC_DIR}/solver" -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="${SP_DIR}" \
  -DCMAKE_PREFIX_PATH="${PREFIX}" \
  -DPROXGQP_VENDOR_DEPENDENCIES=OFF \
  -DPython_EXECUTABLE="${PYTHON}"

cmake --build build --target _core --parallel "${CPU_COUNT}"
cmake --install build --component Unspecified

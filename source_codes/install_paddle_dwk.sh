function prepare()
{
## prepare
sh  ./paddle/internals/scripts/build_scripts/install_deps.jumbo.sh
export PATH=/opt/compiler/gcc-4.8.2/bin/:$PATH

install_path=`pwd`/../
mkl_path=$install_path/paddle/internals/mkl/
cmake -DCMAKE_INSTALL_PREFIX="$install_path" \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DCUDNN_ROOT="${JUMBO_ROOT}/opt/cudnn/" \
    -DMKL_ROOT="$mkl_path" \
    -DWITH_SWIG_PY=ON ..

}

function make_install()
{
make #>/dev/null
#ARGS="-R test_PyDataProvider2 -V"
#make test  
make install
echo 'export PATH=/home/disk5/work5/daiwenkai/baidu/idl/paddle/bin/:$PATH' >> ~/.bashrc
}

function main()
{
mkdir -p build
cd build
#prepare
make_install

}

main 2>&1

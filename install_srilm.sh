#!/bin/bash
VER=1.10
if [ ! -f liblbfgs-$VER.tar.gz ]; then
  wget https://github.com/downloads/chokkan/liblbfgs/liblbfgs-$VER.tar.gz
fi
tar -xzf liblbfgs-$VER.tar.gz
cd liblbfgs-$VER
./configure --prefix=`pwd`
make
make -i install
wd=`pwd`
echo "export LIBLBFGS=$wd/liblbfgs-1.10"
echo export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}':'${LIBLBFGS}'/lib/.libs
mkdir -p srilm
cd srilm
tar -xvzf ../srilm.tgz
if [ $major -le 1 ] && [ $minor -le 7 ] && [ $micro -le 1 ]; then
  echo "Detected version 1.7.1 or earlier. Applying patch."
  patch -p0 < ../extras/srilm.patch
fi

# set the SRILM variable in the top-level Makefile to this directory.
cp Makefile tmpf

cat tmpf | awk -v pwd=`pwd` '/SRILM =/{printf("SRILM = %s\n", pwd); next;} {print;}' \
  > Makefile || exit 1;

mtype=`sbin/machine-type`

echo HAVE_LIBLBFGS=1 >> common/Makefile.machine.$mtype
grep ADDITIONAL_INCLUDES common/Makefile.machine.$mtype | \
    sed 's|$| -I$(SRILM)/../liblbfgs-1.10/include|' \
    >> common/Makefile.machine.$mtype

grep ADDITIONAL_LDFLAGS common/Makefile.machine.$mtype | \
    sed 's|$| -L$(SRILM)/../liblbfgs-1.10/lib/ -Wl,-rpath -Wl,$(SRILM)/../liblbfgs-1.10/lib/|' \
    >> common/Makefile.machine.$mtype
make || exit 1
cd ..
wd=`pwd`
echo "export SRILM=$wd/srilm"
  dirs="\${PATH}"
  for directory in $(cd srilm && find bin -type d ) ; do
    dirs="$dirs:\${SRILM}/$directory"
  done
echo "export PATH=$dirs"

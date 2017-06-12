#!/bin/bash
#!/bin/bash


####### installing liblbfgs ########
VER=1.10
if [ ! -f liblbfgs-$VER.tar.gz ]; then
  wget https://github.com/downloads/chokkan/liblbfgs/liblbfgs-$VER.tar.gz
fi

tar -xzf liblbfgs-$VER.tar.gz
cd liblbfgs-$VER
./configure --prefix=`pwd`
make
# due to the liblbfgs project directory structure, we have to use -i
# but the erros are completely harmless
make -i install
cd ..

(
 
  wd=`pwd`
  echo "export LIBLBFGS=$wd/liblbfgs-1.10"
  echo export LD_LIBRARY_PATH='${LD_LIBRARY_PATH:-}':'${LIBLBFGS}'/lib/.libs
)


######## installing srilm #######

if [ ! -f srilm.tgz ]; then
  echo This script cannot install SRILM in a completely automatic
  echo way because you need to put your address in a download form.
  echo Please download SRILM from http://www.speech.sri.com/projects/srilm/download.html
  echo put it in ./srilm.tgz, then run this script.
  exit 1
fi

! which gawk 2>/dev/null && \
   echo "GNU awk is not installed so SRILM will probably not work correctly: refusing to install" && exit 1;

mkdir -p srilm
cd srilm
tar -xvzf ../srilm.tgz

major=`awk -F. '{ print $1 }' RELEASE`
minor=`awk -F. '{ print $2 }' RELEASE`
micro=`awk -F. '{ print $3 }' RELEASE`

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
(
  
  wd=`pwd`
  echo "export SRILM=$wd/srilm"
  dirs="\${PATH}"
  for directory in $(cd srilm && find bin -type d ) ; do
    dirs="$dirs:\${SRILM}/$directory"
  done
  echo "export PATH=$dirs"
) 
echo >&2 "Installation of SRILM finished successfully"

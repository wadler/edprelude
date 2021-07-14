#!/bin/sh
# A script for installing files corresponding to EDPrelude to your system.

if [ `whoami` != root ]; then
  echo This script needs to be run as sudo in order to install correctly.
  exit
fi

# Install GenericPretty Through Cabal
echo Installing GenericPretty Dependency...
cabal update
cabal install GenericPretty

# EDPrelude information is located in /usr/local/share/edprelude
[ -d /usr/local/share ] || mkdir -p /usr/local/share

#echo Creating /usr/local/share/eprelude
cp -r -p -i edprelude /usr/local/share

#echo Creating /usr/local/bin/edhci
cp -r -p -i edhc/edhci /usr/local/bin

echo Install Complete.

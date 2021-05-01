#!/bin/sh
# A script for removing files corresponding to EDPrelude from your system.

if [ `whoami` != root ]; then
  echo This script needs to be run as sudo in order to uninstall correctly.
  exit
fi


# EDPrelude information is located in /usr/local/share/edprelude and /usr/local/bin/edhci
#echo Deleting /usr/local/share/edprelude/Text/PrettyPrint/GenericPretty.hs...
rm -f -i /usr/local/share/edprelude/Text/PrettyPrint/GenericPretty.hs
#echo Deleting /usr/local/share/edprelude/Text/PrettyPrint...
rm -d -f -i /usr/local/share/edprelude/Text/PrettyPrint
#echo Deleting /usr/local/share/edprelude/Text...
rm -d -f -i /usr/local/share/edprelude/Text
#echo Deleting /usr/local/share/edprelude/.ghci
rm -f -i /usr/local/share/edprelude/.ghci
#echo Deleting /usr/local/share/edprelude/EdPrelude.hs
rm -f -i /usr/local/share/edprelude/EdPrelude.hs
#echo Deleting /usr/local/share/edprelude...
rm -d -f -i /usr/local/share/edprelude
#echo Deleting /usr/local/bin/edhci
rm -f -i /usr/local/bin/edhci
echo Uninstall Complete.

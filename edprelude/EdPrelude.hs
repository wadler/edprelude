{-# LANGUAGE Trustworthy #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
{-# LANGUAGE DeriveGeneric, DeriveAnyClass #-}

-----------------------------------------------------------
-- Module      :  EdPrelude
-- Copyright   :  (c) Matthew Marsland
-- License     :  BSD-style
--
-- Maintainer  :  marslandm@me.com
-- Status      :  work in progress
--
-- The EdPrelude: A version of Prelude built for students from the ground up supporting:
--  - Restricted Numeric Classes
--  - Defaulted Function Signature (Minimized typeclasses to Num (Real, Fractional, Integral), Eq, Ord, Enum, Show, Generic, Out, Applicative, and Functor)
-----------------------------------------------------------

module EdPrelude (
    --EXPORT LIST
    --Numeric Types and Functions
    Num((+),(-),(*),negate,abs,signum,fromInteger),
    Real(toRational), 
    Fractional((/), recip),
    RealFrac(truncate,round,ceiling,floor),
    Integral(quot, rem, div, mod, quotRem, divMod, toInteger),
    Integer, Double, Rational,
    numerator, denominator,
    (%), (^), (^^),
    even, odd,

    --Ordering Types and Functions
    Ord(compare,(<),(<=),(>),(>=),max,min),
    Ordering,

    --Equality and Booleans
    Eq((==),(/=)),
    Bool(True,False), 
    otherwise, (||), (&&), not,

    --Enum
    E.Enum,
    EdPrelude.toEnum, EdPrelude.fromEnum,

    --List Types and Functions
    length, genericLength, take, drop, takeWhile, dropWhile, sum, product, and, or, all, any, (!!), 
    zip, zipWith, unzip, isPrefixOf, map, elem, filter, (++),
    repeat, replicate, cycle, head, tail, init, last, concat, delete, maximum, minimum,
    reverse, sort, sortOn, sortBy, nub, nubBy, unlines, lines, unwords, words, concatMap, null, lookup, transpose,
    foldr, foldr1, foldl, foldl1,
    (\\),
    --Tuple Functions
    fst, snd, curry, uncurry,

    --Char Types and Functions
    isDigit, isUpper, isLower, isAlpha, isAlphaNum, toUpper, toLower, digitToInteger, integerToDigit, chr, ord,

    --Showing as Strings, Characters, and Strings
    Show(..),
    Char, String,

    --Miscellaneous
    undefined, error, errorWithoutStackTrace, ($), (.), seq, sequence_, sequence, id, break,

    --Lifting and Monads
    Monad, Applicative,
    (>>=), (>>), guard, return,
    liftM, liftM2, replicateM,
    Maybe(Just, Nothing),
    Functor, (<$>),

    --IO
    IO,
    putStr, putStrLn, readFile,

    --Random
    randomR, randomRIO, newStdGen,

    --Pretty-Printing
    Generic, Out,
    pp, print
    ) where

-- IMPORTED DEFINITIONS (UNCHANGED):
import GHC.Num
import GHC.Real
import Data.Eq
import Data.Ord
import Data.Bool (otherwise, not, (||), (&&), Bool(True, False))
import Data.Char (isDigit, isUpper, isLower, isAlpha, isAlphaNum)
import qualified GHC.Enum as E (Enum,toEnum,fromEnum)
import qualified Data.Char as C (ord,chr)
import Data.List ((++),takeWhile,dropWhile,delete,reverse,map,filter,zip,unzip,zipWith,
                  isPrefixOf,head,tail,init,last,concat,words,unwords,lines,unlines,transpose,
                  repeat,replicate,sort,sortOn,sortBy,nub,nubBy,lookup,genericLength,(\\))
import Data.Tuple (fst, snd, curry, uncurry)
import GHC.Show
import GHC.Types (Char,Bool(True,False),Double)
import GHC.Base (String,error,errorWithoutStackTrace,($),(.),undefined,seq,id)
import Data.Maybe (Maybe(Just,Nothing))
--import Control.Functor (Functor, (<$>))
import Prelude (Functor,(<$>),sequence_,sequence,cycle,break)
import Control.Monad (Monad,liftM,liftM2,(>>=),(>>),guard,return)
import Control.Applicative (Applicative)
import qualified Control.Monad as M (replicateM)
import System.IO (IO,putStr,putStrLn,readFile)
import System.Random (randomR,randomRIO,newStdGen)
import Text.PrettyPrint
import Text.PrettyPrint.GenericPretty

-- REDEFINED FUNCTIONS:
-- (unmodified type signatures are shown by preceding comments for comparison)

-- **Enum Functions**
--toEnum :: Enum a => Int -> a
toEnum :: E.Enum a => Integer -> a
toEnum i = E.toEnum (fromIntegral i)

--fromEnum :: Enum a => a -> Int
fromEnum :: E.Enum a => a -> Integer
fromEnum x = toInteger (E.fromEnum x)

-- **Char Functions**
--toUpper depends on the modified toEnum and fromEnum, otherwise unchanged
toUpper :: Char -> Char
toUpper c
    | isLower c = toEnum (fromEnum c - fromEnum 'a' + fromEnum 'A')
    | otherwise = c

--toLower depends on the modified toEnum and fromEnum, otherwise unchanged
toLower :: Char -> Char
toLower c
    | isUpper c = toEnum (fromEnum c - fromEnum 'A' + fromEnum 'a')
    | otherwise = c

--digitToInt :: Char -> Int
digitToInteger :: Char -> Integer
digitToInteger c 
    | isDigit c = fromEnum c - fromEnum '0'
    | otherwise = errorWithoutStackTrace "Char.digitToInteger: not a digit."

--intToDigit :: Int -> Char
integerToDigit :: Integer -> Char
integerToDigit i
    | i >= 0 && i <= 9 = toEnum (i + fromEnum '0')
    | otherwise = errorWithoutStackTrace "Char.integerToDigit: not a digit."

--ord :: Char -> Int
ord :: Char -> Integer
ord c = fromEnum c

--chr :: Int -> Char
chr :: Integer -> Char
chr i = toEnum i

-- **List Functions**
--length :: Foldable t => t a -> Int
length :: [a] -> Integer
length [] = 0
length (x:xs) = 1 + (length xs)

--take :: Int -> [a] -> [a]
take :: Integer -> [a] -> [a]
take n [] = []
take 0 (x:xs) = []
take n (x:xs) = x : take (n-1) xs

--drop :: Int -> [a] -> [a]
drop :: Integer -> [a] -> [a]
drop n [] = []
drop 0 xs = xs
drop n (x:xs) = drop (n-1) xs

--sum :: (Num a, Foldable t) => t a -> a
sum :: Num a => [a] -> a
sum [] = 0
sum (x:xs) = x + sum xs

--product :: (Num a, Foldable t) => t a -> a
product :: Num a => [a] -> a
product [] = 1
product (x:xs) = x * product xs

--and :: Foldable t => t Bool -> Bool
and :: [Bool] -> Bool
and [] = True
and (b:bs) = b && (and bs)

--or :: Foldable t => t Bool -> Bool
or :: [Bool] -> Bool
or [] = False
or (b:bs) = b || (or bs)

--all :: Foldable t => (a -> Bool) -> t a -> Bool
all :: (a -> Bool) -> [a] -> Bool
all f xs = and (map f xs)

--any :: Foldable t => (a -> Bool) -> t a -> Bool
any :: (a -> Bool) -> [a] -> Bool
any f xs = or (map f xs)

--maximum :: (Ord a, Foldable t) => t a -> a
maximum :: (Ord a) => [a] -> a
maximum [] = errorWithoutStackTrace "EdPrelude.maximum: empty list"
maximum [x] = x
maximum (x:xs) = max x (maximum xs)

--minimum :: (Ord a, Foldable t) => t a -> a
minimum :: (Ord a) => [a] -> a
minimum [] = errorWithoutStackTrace "EdPrelude.minimum: empty list"
minimum [x] = x
minimum (x:xs) = min x (minimum xs)

--(!!) :: [a] -> Int -> a
infixl 9 !!
(!!) :: [a] -> Integer -> a
xs     !! n | n < 0 =  errorWithoutStackTrace "EdPrelude.!!: negative index"
[]     !! _         =  errorWithoutStackTrace "EdPrelude.!!: index too large"
(x:_)  !! 0         =  x
(_:xs) !! n         =  xs !! (n-1)

--concatMap :: Foldable t => (a -> [b]) -> t a -> [b]
concatMap :: (a -> [b]) -> [a] -> [b]
concatMap f as = concat (map f as)

--elem :: (Eq a, Foldable t) => a -> t a -> Bool
elem :: (Eq a) => a -> [a] -> Bool
elem x [] = False
elem x (y:ys) = x == y || elem x ys

--null :: Foldable t => t a -> Bool
null :: [a] -> Bool
null [] = True
null xs = False

--foldr :: Foldable t => (a -> b -> b) -> b -> t a -> b
foldr :: (a -> b -> b) -> b -> [a] -> b
foldr _ v [] = v
foldr f v xs = f (head xs) (foldr f v (tail xs))

--foldl :: Foldable t => (a -> b -> a) -> b -> t a -> b
foldl :: (a -> b -> a) -> a -> [b] -> a
foldl _ v [] = v
foldl f v xs = f (foldl f v (init xs)) (last xs)

--foldr1 :: Foldable t => (a -> a -> a) -> t a -> a
foldr1 :: (a -> a -> a) -> [a] -> a
foldr1 _ [] = errorWithoutStackTrace "EdPrelude.foldr1: empty list"
foldr1 _ [x] = x
foldr1 f xs = f (head xs) (foldr1 f (tail xs))

--foldl1 :: Foldable t => (a -> a -> a) -> t a -> a
foldl1 :: (a -> a -> a) -> [a] -> a
foldl1 _ [] = errorWithoutStackTrace "EdPrelude.foldl1: empty list"
foldl1 _ [x] = x
foldl1 f xs = f (foldl1 f (init xs)) (last xs)

-- **Monad Functions**
--replicateM :: Applicative m => Int -> m a -> m [a]
replicateM :: Applicative m => Integer -> m a -> m [a]
replicateM i m = M.replicateM (fromIntegral i) m

-- **Pretty-Printing Functions**
print           :: Out a => a -> IO ()
print x         =  ppStyle (Style {mode = PageMode, lineLength = 80, ribbonsPerLine = 2}) x

-- Automatic Derivation of Out Instances from Show Instances
instance {-# OVERLAPPABLE #-} Show a => Out a where
    doc x = text (show x)
    docPrec _ x = doc x


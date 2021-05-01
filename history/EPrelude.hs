{-# LANGUAGE Trustworthy #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE FlexibleInstances #-}
{-# LANGUAGE UndecidableInstances #-}
--{-# LANGUAGE OverlappingInstances #-}

-----------------------------------------------------------
-- Module      :  EPrelude
-- Copyright   :  (c) Matthew Marsland
-- License     :  BSD-style
--
-- Maintainer  :  marslandm@me.com
-- Status      :  work in progress
--
-- The EPrelude: A modification of Prelude specifically restricting Numeric classes,
-- and altering type signatures accordingly.
-----------------------------------------------------------

module EPrelude (

    -- # Standard types, classes and related functions

    -- ## Basic data types
    Bool(False, True),
    (&&), (||), not, otherwise,

    Maybe(Nothing, Just),
    maybe,

    Either(Left, Right),
    either,

    Ordering(LT, EQ, GT),
    Char, String,

    -- ### Tuples
    fst, snd, curry, uncurry,

    -- ## Basic type classes
    Eq((==), (/=)),
    Ord(compare, (<), (<=), (>=), (>), max, min),
    --Enum(succ, pred, toEnum, fromEnum, enumFrom, enumFromThen, enumFromTo, enumFromThenTo),
    Enum(succ, pred, enumFrom, enumFromThen, enumFromTo, enumFromThenTo),
    EPrelude.fromEnum, EPrelude.toEnum,
    Bounded(minBound, maxBound),

    -- ## Numbers

    -- ### Numeric types
    Integer, Float, Double, Rational,
    --Int, Integer, Float, Double,
    --Rational, Word,

    -- ### Numeric type classes
    Num((+), (-), (*), negate, abs, signum, fromInteger),
    Real(toRational),
    Integral(quot, rem, div, mod, quotRem, divMod, toInteger),
    Fractional((/), recip, fromRational),
    Floating(pi, exp, log, sqrt, (**), logBase, sin, cos, tan,
             asin, acos, atan, sinh, cosh, tanh, asinh, acosh, atanh),
    RealFrac(properFraction, truncate, round, ceiling, floor),
    RealFloat(floatRadix, floatDigits, floatRange, decodeFloat,
              encodeFloat, exponent, significand, scaleFloat, isNaN,
              isInfinite, isDenormalized, isIEEE, isNegativeZero, atan2),

    -- ### Numeric functions
    subtract, even, odd, gcd, lcm, (^), (^^),
    fromIntegral, realToFrac,

    -- ## Semigroups and Monoids
    Semigroup((<>)),
    Monoid(mempty, mappend, mconcat),

    -- ## Monads and functors
    Functor(fmap, (<$)), (<$>),
    Applicative(pure, (<*>), (*>), (<*)),
    Monad((>>=), (>>), return),
    MonadFail(fail),
    mapM_, sequence_, (=<<),

    {-
    ---- ## Folds and traversals
    Foldable(elem,      -- :: (Foldable t, Eq a) => a -> t a -> Bool
             -- fold,   -- :: Monoid m => t m -> m
             foldMap,   -- :: Monoid m => (a -> m) -> t a -> m
             foldr,     -- :: (a -> b -> b) -> b -> t a -> b
             -- foldr', -- :: (a -> b -> b) -> b -> t a -> b
             foldl,     -- :: (b -> a -> b) -> b -> t a -> b
             -- foldl', -- :: (b -> a -> b) -> b -> t a -> b
             foldr1,    -- :: (a -> a -> a) -> t a -> a
             foldl1,    -- :: (a -> a -> a) -> t a -> a
             maximum,   -- :: (Foldable t, Ord a) => t a -> a
             minimum,   -- :: (Foldable t, Ord a) => t a -> a
             product,   -- :: (Foldable t, Num a) => t a -> a
             sum),      -- :: Num a => t a -> a
             -- toList) -- :: Foldable t => t a -> [a]
    -}
    elem, {-foldMap,-} foldr, foldl, foldr1, foldl1, maximum, minimum, product, sum,

    Traversable(traverse, sequenceA, mapM, sequence),

    -- ## Miscellaneous functions
    id, const, (.), flip, ($), until,
    asTypeOf, error, errorWithoutStackTrace, undefined,
    seq, ($!),

    -- # List operations
    List.map, (List.++), List.filter,
    --List.head, List.last, List.tail, List.init, (List.!!),
    List.head, List.last, List.tail, List.init, (!!),
    {-Foldable.null,-} null, length,
    List.reverse,
    -- ### Special folds
    {-Foldable.and, Foldable.or, Foldable.any, Foldable.all,
    Foldable.concat, Foldable.concatMap,-}
    and, or, any, all, concat, concatMap,
    -- ## Building lists
    -- ### Scans
    List.scanl, List.scanl1, List.scanr, List.scanr1,
    -- ### Infinite lists
    --List.iterate, List.repeat, List.replicate, List.cycle,
    List.iterate, List.repeat, replicate, List.cycle,
    -- ## Sublists
    --List.take, List.drop,
    take, drop,
    List.takeWhile, List.dropWhile,
    List.span, List.break,
    --List.splitAt,
    splitAt,
    -- ## Searching lists
    {-Foldable.notElem,-} List.notElem,
    List.lookup,
    -- ## Zipping and unzipping lists
    List.zip, List.zip3,
    List.zipWith, List.zipWith3,
    List.unzip, List.unzip3,
    -- ## Functions on strings
    List.lines, List.words, List.unlines, List.unwords,

    -- # Converting to and from @String@
    -- ## Converting to @String@
    ShowS,
    --Show(showsPrec, showList, show),
    Show(showList, show),
    EPrelude.showsPrec,
    shows,
    showChar, showString, showParen,
    -- ## Converting from @String@
    ReadS,
    --Read(readsPrec, readList),
    Read(readList),
    EPrelude.readsPrec,
    reads, readParen, read, lex,

    -- # Basic Input and output
    IO,
    -- ## Simple I\/O operations
    -- All I/O functions defined here are character oriented.  The
    -- treatment of the newline character will vary on different systems.
    -- For example, two characters of input, return and linefeed, may
    -- read as a single newline character.  These functions cannot be
    -- used portably for binary I/O.
    -- ### Output functions
    putChar,
    putStr, putStrLn, print,
    -- ### Input functions
    getChar,
    getLine, getContents, interact,
    -- ### Files
    FilePath,
    readFile, writeFile, appendFile, readIO, readLn,
    -- ## Exception handling in the I\/O monad
    IOError, ioError, userError,

    -- # Generic Pretty-Printing
    Generic, Out, pp

  ) where

import Control.Monad
import Control.Monad.Fail
import System.IO        hiding (print)
import System.IO.Error
import qualified Data.List as List
import Data.Either
--import Data.Foldable    ( Foldable(..) )
--import qualified Data.Foldable as Foldable
import Data.Functor     ( (<$>) )
import Data.Maybe
import Data.Traversable ( Traversable(..) )
import Data.Tuple
import Data.Semigroup

import GHC.Base hiding ( foldr, mapM, sequence )
import Text.Read
import GHC.Enum
import GHC.Num
import GHC.Real
import GHC.Float
import GHC.Show

import Text.PrettyPrint
import Text.PrettyPrint.GenericPretty
import GHC.Generics

-- # Alternative definitions (INTEGER):

-- ## LENGTH
length :: [a] -> Integer
length [] = 0
length (x:xs) = 1 + (length xs)

r_length :: [a] -> Integer
r_length = fromIntegral . List.length

-- ## TAKE
take :: Integer -> [a] -> [a]
take n [] = []
take 0 (x:xs) = []
take n (x:xs) = x : take (n-1) xs

r_take :: Integer -> [a] -> [a]
r_take n xs = List.take (fromIntegral n) xs

-- ## DROP
drop :: Integer -> [a] -> [a]
--drop n xs = List.drop (fromIntegral n) xs
drop = n_drop

n_drop :: Integer -> [a] -> [a]
n_drop n [] = []
n_drop 0 xs = xs
n_drop n (x:xs) = n_drop (n-1) xs

-- ## REPLICATE
replicate :: Integer -> a -> [a]
--replicate n x = List.replicate (fromIntegral n) x
replicate = n_replicate

n_replicate :: Integer -> a -> [a]
n_replicate n x = take n (List.repeat x)

-- ## SPLITAT
splitAt :: Integer -> [a] -> ([a], [a])
--splitAt n xs = List.splitAt (fromIntegral n) x
splitAt = n_splitAt

n_splitAt :: Integer -> [a] -> ([a], [a])
n_splitAt n xs = (take n xs, drop n xs)

-- ## (!!)
infixl 9 !!

-- ### (!!) CAST
{-
(!!) :: [a] -> Integer -> Maybe a
(!!) xs n = (List.!!) xs (fromIntegral n)
--}

-- ### (!!) NAIVE
--{-
(!!) :: [a] -> Integer -> a
xs     !! n | n < 0 =  errorWithoutStackTrace "EPrelude.!!: negative index"
[]     !! _         =  errorWithoutStackTrace "EPrelude.!!: index too large"
(x:_)  !! 0         =  x
(_:xs) !! n         =  xs !! (n-1)
--}

-- ### (!!) MAYBE 
{-
(!!) :: [a] -> Integer -> Maybe a
xs     !! n | n < 0 =  Nothing
[]     !! _         =  Nothing
(x:_)  !! 0         =  Just x
(_:xs) !! n         =  xs !! (n-1)
--}

-- # ENUM
-- ## FROMENUM
fromEnum :: Enum a => a -> Integer
fromEnum = fromIntegral . GHC.Enum.fromEnum

-- ## TOENUM
toEnum :: Enum a => Integer -> a
toEnum n = GHC.Enum.toEnum (fromIntegral n)

-- # SHOW
-- ## SHOWSPREC
showsPrec :: Show a => Integer -> a -> ShowS
showsPrec n = GHC.Show.showsPrec (fromIntegral n)

-- # READ
-- ## READSPREC
readsPrec :: Read a => Integer -> ReadS a
readsPrec n = Text.Read.readsPrec (fromIntegral n)


-- # Alternative definitions (FOLDABLE):
elem :: Eq a => a -> [a] -> Bool
elem _ [] = False
elem k (x:xs)   | k == x = True
                | otherwise = elem k xs

notElem :: Eq a => a -> [a] -> Bool
notElem x = not . elem x

foldr :: (a -> b -> b) -> b -> [a] -> b
foldr _ v [] = v
foldr f v xs = f (List.head xs) (foldr f v (List.tail xs))

foldl :: (a -> b -> a) -> a -> [b] -> a
foldl _ v [] = v
foldl f v xs = f (foldl f v (List.init xs)) (List.last xs)

foldr1 :: (a -> a -> a) -> [a] -> a
foldr1 _ [] = errorWithoutStackTrace "EPrelude.foldr1: empty list"
foldr1 _ [x] = x
foldr1 f xs = f (List.head xs) (foldr1 f (List.tail xs))

foldl1 :: (a -> a -> a) -> [a] -> a
foldl1 _ [] = errorWithoutStackTrace "EPrelude.foldl1: empty list"
foldl1 _ [x] = x
foldl1 f xs = f (foldl1 f (List.init xs)) (List.last xs)

maximum :: Ord a => [a] -> a
maximum [] = errorWithoutStackTrace "EPrelude.minimum: empty list"
maximum xs = foldr1 max xs

minimum :: Ord a => [a] -> a
minimum [] = errorWithoutStackTrace "EPrelude.minimum: empty list"
minimum xs = foldr1 min xs

product :: Num a => [a] -> a
product xs = foldr (*) 1 xs

sum :: Num a => [a] -> a
sum xs = foldr (+) 0 xs

and :: [Bool] -> Bool
and = foldr (&&) True

or :: [Bool] -> Bool
or = foldr (||) False

any :: (a -> Bool) -> [a] -> Bool
any f xs = or (map f xs)

all :: (a -> Bool) -> [a] -> Bool
all f xs = and (map f xs)

null :: [a] -> Bool
null [] = True
null xs = False

concat :: [[a]] -> [a]
concat = foldr (++) []

concatMap :: (a -> [b]) -> [a] -> [b]
concatMap f xs = concat (map f xs)

-- # Alternative definitions (PRINTING):
print           :: Out a => a -> IO ()
print x         =  ppStyle (Style {mode = PageMode, lineLength = 80, ribbonsPerLine = 2}) x

-- # Automatic Derivation of Out Instances from Show Instances
--instance {-{-# OVERLAPPABLE #-}-} (Show a) => Out (a) where
--    doc x = text (show x)
--    docPrec _ = doc
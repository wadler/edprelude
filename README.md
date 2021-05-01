# Ed(inburgh|ucational) Prelude
An alternative Haskell Prelude that supports automatic pretty-printing through GenericPretty, improved numeric type safety and function signature intelligibility.
Primarily an experiment to see what the potential performance and readability impacts are.

## Motivation
To improve the approachability and readability of Haskell code/data. New users can stumble over the `Foldable` type signatures of functions they only ever apply to lists, over the subtle difference between Int and Integer, over the unreadability of Node (Node (Leaf 1) (Node (Leaf 2) (Leaf 3))) (Leaf 4). This project attempts to analyze different alterations to the core Prelude library, imported by default in all Haskell programs, aimed at addressing these problems to make jumping into Haskell as easy as it can get.

## Installation
Unfortunately only Unix installations are available for the moment. To install, download the repository above, unzip it, and run the installation script (requires sudo). Then, try it out by navigating to the `examples/tutorial1` folder and running `edhci Tutorial1`. Full instructions are present [here](https://matthewmarmalade.github.io/e-prelude-pages/).

## Usage
Any Haskell project you would like to use EPrelude for requires an 'import EPrelude' statement at the top. From here, make sure you load the file with `edhci` instead of `ghci`. That's it!

## Features
### Automatic Pretty-Printing
The `print` function is redefined to refer to the Generic Pretty-Printer developed by Razvan Ranca, and is immediately accessible via an alternative REPL command. The pretty-printer is automatically derived for any new types with the inclusion of `deriving (Generic, Out)` after the type declaration. However, this does not break any code relying on Show - all instances of Show automatically guarantee single-line instances of the pretty-printer. The REPL's printer has only gotten prettier - not more selective!

### Clear (If Naive) Definitions
EPrelude will provide a flat-file, straightforward implementation of many common Prelude functions that are normally invisible to programmer and reliant on advanced Haskell operations. While Prelude compiles into an efficient standard library, and its definitions generally make use of extremely cool Haskell, it's not ideal for the Inf1A-FP students learning the immediate fundamentals of programming. EPrelude will address this by exposing every definition of what it exports, in simple terms familiar to new Haskellers from the way they learn about them; prioritizing readability over optimization for short-term learning.

### Int -> Integer
Int is no longer accessible as a type (if you need it, GHC.Types.Int is still available), because it is merely a bounded form of Integer. All functions with Ints in their type signatures have been replaced with versions using Integer instead.

### Foldable t -> [a]
The ability to apply fold operations to a wide class of types is a very powerful Haskell feature, however it is confusing to new users to see a type signature like `length :: Foldable t => t a -> Int`. This Prelude, aimed at said new users, gives them what they expect: `length :: [a] -> Integer`.

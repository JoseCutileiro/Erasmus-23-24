import Data.List
import System.IO

fib = 1 : 1 : [a + b | (a,b) <- zip fib (tail fib)]

main = do
    print (take 100 fib)
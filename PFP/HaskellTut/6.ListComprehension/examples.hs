import Data.List
import System.IO


main = do

    -- change values
    print [3^n | n <- [1..10]]

    -- list opeations (mult table)
    print [[x * y | y <-[1..10]] | x <- [1..10]]
import Data.List
import System.IO


main = do

    -- Update the values on the list
    print [x * 2 | x <- [1..10]]

    -- Filter
    print [x | x <- [1..10], x <= 5 && x > 1]

    -- Combine update and filter
    print [x * 3 | x <- [1..10], x * 3 < 15]

    -- mod filter
    print [x | x <- [1..500], mod x 13 == 0]
    print [13,26..500]

    -- sort
    print (sort [9,2,5,0,1])

    -- sum lists (free op)
    print (zipWith (+) [1,2,3,4,5] [5,4,3,2,1])

    -- other filter
    print (filter (>5) [1,3,5,7,9,11])

    -- takeWhile
    print (takeWhile (<= 20) [2,4..])

    -- foldl (or foldr) (free op)
    print (foldl (+) 2 [2,3,4,5])

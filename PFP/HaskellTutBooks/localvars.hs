lend :: Int -> Int -> Int

lend initial final = if (final <= 0)
                        then 0
                    else
                        amount
        where 
            amount = final - initial
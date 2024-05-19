let prefix_sum [n] (arr: [n]i32): [n]i32 =
  let indices = iota n
  in map (\i ->
           if i == 0 then arr[0]
           else arr[i] + arr[i-1]) indices

let pfp [n] (arr: [n]i32): i32 =
    let temp = prefix_sum arr
    in temp[n-1]
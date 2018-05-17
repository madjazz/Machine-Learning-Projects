package main

import (
  "fmt"
  "math"
)

func function(x float64) float64 {

  y := math.Pow(x, 4) - math.Pow(2 * x, 3) + math.Pow(4 * x, 2) - 10

  return y

}

func main () {

  var cur_x float64
  var gamma float64
  var precision float64
  var previous_step_size float64

  cur_x = 6
  gamma = 0.01
  precision = 0.00001
  previous_step_size = cur_x

  for previous_step_size > precision {

    prev_x := cur_x
    cur_x += -gamma * function(prev_x)
    previous_step_size = math.Abs(cur_x - prev_x)
    fmt.Println(previous_step_size)
  }

output := fmt.Sprintf("The local minimum occurs at %f", cur_x)
fmt.Println(output)

}

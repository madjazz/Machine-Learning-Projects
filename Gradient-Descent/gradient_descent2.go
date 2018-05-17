package main

import (
  "fmt"
  "math"
)

func function(x float64) float64{

  y := math.Pow(x, 4) - math.Pow(2 * x, 3) + math.Pow(4 * x, 2) - 10
  return y
}

func Sgn(a float64) int {

  switch {

  case a < 0:
    return -1

  case a > 0:
    return +1
  }

  return 0
}


func main() {

  var n int
  var nmax int

  var c float64
  var a float64
  var b float64
  var tol float64

  nmax = 1000
  a = 0
  b = 100
  tol = 0.0000001

  n = 1

  for n <= nmax {
    c = (a + b) / 2

    if function(c) == 0 || (b - a) / 2 < tol {
      fmt.Println(c)
      break
    } else {
      n = n + 1
      fmt.Println(n)
    }

    if Sgn(function(c)) == Sgn(function(a)) {
      a = c
    } else {
      b = c
    }

  }
}

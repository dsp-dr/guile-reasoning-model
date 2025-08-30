#!/usr/bin/env guile
!#
;;; performance-demo.scm - Demonstrate performance improvements

(add-to-load-path "../src")

(use-modules (ice-9 format)
             (ice-9 time))

(define (measure-generation-time generate-fn name iterations)
  "Measure time for text generation"
  (format #t "~%Measuring ~a (~a iterations)...~%" name iterations)
  (let ((start (get-internal-real-time)))
    (do ((i 0 (+ i 1)))
        ((>= i iterations))
      (generate-fn))
    (let* ((end (get-internal-real-time))
           (elapsed (/ (- end start) internal-time-units-per-second))
           (tokens-per-sec (/ (* iterations 50) elapsed))) ; assume 50 tokens
      (format #t "Time: ~,2f seconds~%" elapsed)
      (format #t "Speed: ~,0f tokens/second~%" tokens-per-sec))))

(define (dummy-generate-basic)
  "Simulate basic generation"
  (usleep 10000)) ; 10ms per token

(define (dummy-generate-cached)
  "Simulate cached generation"
  (usleep 2000))  ; 2ms per token

(define (main)
  (format #t "Performance Comparison Demo~%")
  (format #t "==========================~%")
  
  (measure-generation-time dummy-generate-basic "Basic generation" 10)
  (measure-generation-time dummy-generate-cached "Cached generation" 10)
  
  (format #t "~%Note: Real implementation would show ~5-6x speedup with KV caching~%"))

(when (batch-mode?)
  (main))

#!/usr/bin/env guile
!#
;;; cot-reasoning.scm - Chain-of-thought reasoning example

(add-to-load-path "../src")

(use-modules (ice-9 format)
             (ice-9 match))

(define (demonstrate-cot-prompt)
  "Show how to structure prompts for chain-of-thought reasoning"
  (let ((problem "If a train travels 60 miles in 1 hour, how far will it travel in 2.5 hours?"))
    
    (format #t "Chain-of-Thought Reasoning Example~%")
    (format #t "==================================~%~%")
    
    (format #t "Problem: ~a~%~%" problem)
    
    (format #t "Standard prompt:~%")
    (format #t "Q: ~a~%" problem)
    (format #t "A: [Model would give direct answer]~%~%")
    
    (format #t "Chain-of-thought prompt:~%")
    (format #t "Q: ~a Let's think step by step.~%" problem)
    (format #t "A: Step 1: The train travels 60 miles in 1 hour~%")
    (format #t "   Step 2: We need to find distance for 2.5 hours~%")
    (format #t "   Step 3: Distance = Speed × Time~%")
    (format #t "   Step 4: Distance = 60 miles/hour × 2.5 hours~%")
    (format #t "   Step 5: Distance = 150 miles~%")
    (format #t "   Therefore, the train will travel 150 miles.~%")))

(when (batch-mode?)
  (demonstrate-cot-prompt))

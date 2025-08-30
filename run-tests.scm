#!/usr/bin/env guile
!#
;;; run-tests.scm - Test runner

(add-to-load-path "src")
(use-modules (srfi srfi-64))

(test-runner-factory
 (lambda ()
   (let ((runner (test-runner-simple)))
     (test-runner-on-final! runner
       (lambda (runner)
         (format #t "~%Test Summary:~%")
         (format #t "Passed: ~a~%" (test-runner-pass-count runner))
         (format #t "Failed: ~a~%" (test-runner-fail-count runner))
         (format #t "Skipped: ~a~%~%" (test-runner-skip-count runner))))
     runner)))

;; Run all test files
(for-each (lambda (test-file)
            (format #t "Running ~a...~%" test-file)
            (load test-file))
          (find-files "tests" ".*\\.scm$"))

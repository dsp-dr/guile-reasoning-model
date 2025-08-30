;;; tokenizer-test.scm - Tokenizer tests

(use-modules (srfi srfi-64)
             (tokenizer base))

(test-begin "tokenizer-base")

(test-assert "Create tokenizer"
  (tokenizer? (make-tokenizer 
               (make-hash-table)
               (lambda (x) '())
               (lambda (x) "")
               (make-hash-table))))

(test-end "tokenizer-base")

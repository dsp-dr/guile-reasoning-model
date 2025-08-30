#!/usr/bin/env guile
!#
;;; basic-generation.scm - Basic text generation example

(add-to-load-path "../src")

(use-modules (tokenizer base)
             (model base)
             (generation basic)
             (ice-9 format))

(define (create-dummy-tokenizer)
  "Create a dummy tokenizer for demonstration"
  (let ((vocab (make-hash-table))
        (reverse-vocab (make-hash-table)))
    ;; Simple word-level tokenization
    (for-each (lambda (pair)
                (hash-set! vocab (car pair) (cdr pair))
                (hash-set! reverse-vocab (cdr pair) (car pair)))
              '(("Hello" . 1) ("world" . 2) ("!" . 3) 
                ("<eos>" . 4) (" " . 5)))
    
    (make-tokenizer 
     vocab
     (lambda (text)
       ;; Very simple tokenization
       (map (lambda (word) 
              (hash-ref vocab word 0))
            (string-split text #\space)))
     (lambda (ids)
       (string-join 
        (map (lambda (id) 
               (hash-ref reverse-vocab id "<??>"))
             ids)
        " "))
     (alist->hash-table '((eos . 4))))))

(define (create-dummy-model)
  "Create a dummy model that returns random logits"
  (make-model
   '((vocab-size . 100)
     (n-layers . 2))
   (lambda (input-ids cache)
     ;; Return random logits
     (let ((seq-len (length input-ids)))
       (vector-unfold (lambda (i)
                       (vector-unfold (lambda (j) 
                                       (random:uniform))
                                     100))
                     seq-len)))
   #f))

(define (main)
  (let ((tokenizer (create-dummy-tokenizer))
        (model (create-dummy-model)))
    
    (format #t "Basic text generation example~%")
    (format #t "=============================~%")
    
    (let ((prompt "Hello world"))
      (format #t "Prompt: ~a~%" prompt)
      (format #t "Generated: ~a~%" 
              (generate-text-basic model tokenizer prompt 
                                  #:max-new-tokens 10
                                  #:eos-token-id 4)))))

(when (batch-mode?)
  (main))

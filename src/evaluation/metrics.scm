;;; evaluation/metrics.scm - Chapter 3: Evaluation metrics for reasoning models
(define-module (evaluation metrics)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-43)
  #:use-module (ice-9 match)
  #:export (perplexity
            bleu-score
            rouge-score
            exact-match
            f1-score
            reasoning-accuracy
            chain-accuracy))

;; 3.1 Perplexity
(define (perplexity logits targets)
  "Calculate perplexity from model logits and target tokens"
  (let* ((losses (map cross-entropy logits targets))
         (avg-loss (/ (apply + losses) (length losses))))
    (exp avg-loss)))

(define (cross-entropy logits target)
  "Cross-entropy loss for single prediction"
  (- (log (vector-ref (softmax logits) target))))

(define (softmax vec)
  "Apply softmax to vector"
  (let* ((max-val (vector-fold max -inf.0 vec))
         (exp-vec (vector-map (lambda (x) (exp (- x max-val))) vec))
         (sum (vector-fold + 0 exp-vec)))
    (vector-map (lambda (x) (/ x sum)) exp-vec)))

;; 3.2 BLEU Score
(define (bleu-score candidate references #:optional (max-n 4))
  "Calculate BLEU score for generated text"
  (let* ((precisions (map (lambda (n)
                            (n-gram-precision candidate references n))
                          (iota max-n 1)))
         (brevity (brevity-penalty candidate references))
         (geom-mean (exp (/ (apply + (map log precisions)) max-n))))
    (* brevity geom-mean)))

(define (n-gram-precision candidate references n)
  "Calculate n-gram precision"
  (let ((cand-ngrams (extract-ngrams candidate n))
        (ref-ngrams (append-map (lambda (r) (extract-ngrams r n)) references)))
    (/ (count-matches cand-ngrams ref-ngrams)
       (max 1 (length cand-ngrams)))))

(define (extract-ngrams tokens n)
  "Extract n-grams from token list"
  (if (< (length tokens) n)
      '()
      (cons (take tokens n)
            (extract-ngrams (cdr tokens) n))))

(define (count-matches cand-ngrams ref-ngrams)
  "Count matching n-grams"
  (fold (lambda (ng count)
          (if (member ng ref-ngrams equal?)
              (+ count 1)
              count))
        0
        cand-ngrams))

(define (brevity-penalty candidate references)
  "Calculate brevity penalty for BLEU"
  (let* ((c-len (length candidate))
         (r-len (apply min (map length references))))
    (if (> c-len r-len)
        1.0
        (exp (- 1 (/ r-len c-len))))))

;; 3.2 ROUGE Score
(define (rouge-score candidate reference #:optional (type 'rouge-1))
  "Calculate ROUGE score"
  (case type
    ((rouge-1) (rouge-n candidate reference 1))
    ((rouge-2) (rouge-n candidate reference 2))
    ((rouge-l) (rouge-l-score candidate reference))
    (else (error "Unknown ROUGE type" type))))

(define (rouge-n candidate reference n)
  "ROUGE-N score"
  (let* ((cand-ngrams (extract-ngrams candidate n))
         (ref-ngrams (extract-ngrams reference n))
         (overlap (count-matches cand-ngrams ref-ngrams)))
    (/ overlap (max 1 (length ref-ngrams)))))

(define (rouge-l-score candidate reference)
  "ROUGE-L using longest common subsequence"
  (let* ((lcs-len (lcs-length candidate reference))
         (precision (/ lcs-len (length candidate)))
         (recall (/ lcs-len (length reference))))
    (if (= (+ precision recall) 0)
        0
        (/ (* 2 precision recall) (+ precision recall)))))

(define (lcs-length seq1 seq2)
  "Longest common subsequence length"
  ;; Simplified LCS implementation
  (cond
   ((or (null? seq1) (null? seq2)) 0)
   ((equal? (car seq1) (car seq2))
    (+ 1 (lcs-length (cdr seq1) (cdr seq2))))
   (else (max (lcs-length (cdr seq1) seq2)
              (lcs-length seq1 (cdr seq2))))))

;; 3.3 Reasoning-specific metrics
(define (exact-match prediction reference)
  "Exact match accuracy"
  (if (equal? prediction reference) 1.0 0.0))

(define (f1-score predictions references)
  "F1 score for token-level accuracy"
  (let* ((true-positives (count-matches predictions references))
         (precision (/ true-positives (max 1 (length predictions))))
         (recall (/ true-positives (max 1 (length references)))))
    (if (= (+ precision recall) 0)
        0
        (/ (* 2 precision recall) (+ precision recall)))))

(define (reasoning-accuracy reasoning-chain expected-steps)
  "Evaluate reasoning chain accuracy"
  (let* ((correct-steps (filter (lambda (step)
                                  (member step expected-steps equal?))
                                reasoning-chain))
         (precision (/ (length correct-steps) 
                      (max 1 (length reasoning-chain))))
         (recall (/ (length correct-steps)
                   (max 1 (length expected-steps)))))
    (f1-score reasoning-chain expected-steps)))

(define (chain-accuracy chains expected)
  "Evaluate chain-of-thought accuracy"
  (/ (fold + 0 (map (lambda (c e) 
                     (if (valid-reasoning? c e) 1 0))
                   chains expected))
     (length chains)))

(define (valid-reasoning? chain expected)
  "Check if reasoning chain is valid"
  ;; Check logical consistency and conclusion validity
  (and (>= (length chain) 2)  ; Has steps
       (equal? (last chain) (last expected))  ; Correct conclusion
       (> (reasoning-accuracy chain expected) 0.5)))  ; Reasonable accuracy
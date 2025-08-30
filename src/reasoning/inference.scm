;;; Inference-time compute scaling for reasoning models
;;; Chapter 4 implementation

(define-module (reasoning inference)
  #:use-module (ice-9 match)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:use-module (reasoning core)
  #:export (self-consistency
            majority-voting
            beam-search-reasoning
            monte-carlo-reasoning
            temperature-sampling))

;;; Self-consistency through multiple sampling

(define* (self-consistency problem reasoning-fn #:key (num-samples 5))
  "Generate multiple reasoning chains and aggregate results."
  (let* ((chains (map (lambda (_) 
                       (chain-of-thought problem reasoning-fn))
                     (iota num-samples)))
         (answers (map get-final-answer chains))
         (majority-answer (find-majority answers)))
    (make-reasoning-chain problem
                         #:steps (aggregate-steps chains)
                         #:answer majority-answer
                         #:confidence (calculate-agreement answers majority-answer))))

(define (find-majority items)
  "Find the most common item in a list."
  (let ((counts (count-occurrences items)))
    (car (fold (lambda (pair best)
                (if (> (cdr pair) (cdr best))
                    pair
                    best))
              (cons #f 0)
              counts))))

(define (count-occurrences items)
  "Count occurrences of each unique item."
  (let ((table (make-hash-table)))
    (for-each (lambda (item)
               (hash-set! table item 
                         (1+ (hash-ref table item 0))))
             items)
    (hash-map->list cons table)))

(define (calculate-agreement answers majority)
  "Calculate agreement ratio for self-consistency."
  (let ((agree-count (count (lambda (a) (equal? a majority)) answers)))
    (/ agree-count (length answers))))

(define (aggregate-steps chains)
  "Aggregate steps from multiple chains."
  (let ((all-steps (append-map get-reasoning-steps chains)))
    (take all-steps (min 10 (length all-steps)))))  ; Limit to 10 steps

;;; Voting and verification strategies

(define (majority-voting candidates)
  "Simple majority voting among candidate answers."
  (find-majority candidates))

(define* (weighted-voting candidates weights)
  "Weighted voting based on confidence scores."
  (let ((weighted-counts (make-hash-table)))
    (for-each (lambda (candidate weight)
               (hash-set! weighted-counts candidate
                         (+ weight (hash-ref weighted-counts candidate 0))))
             candidates weights)
    (car (fold (lambda (pair best)
                (if (> (cdr pair) (cdr best))
                    pair
                    best))
              (cons #f 0)
              (hash-map->list cons weighted-counts)))))

;;; Beam search for reasoning paths

(define-record-type <beam-state>
  (make-beam-state path score)
  beam-state?
  (path beam-state-path)
  (score beam-state-score))

(define* (beam-search-reasoning problem expand-fn score-fn 
                                #:key (beam-width 3) (max-depth 5))
  "Beam search through reasoning paths."
  (let loop ((states (list (make-beam-state (list problem) 0)))
             (depth 0))
    (if (or (>= depth max-depth)
            (null? states))
        (car (sort states (lambda (a b)
                           (> (beam-state-score a)
                              (beam-state-score b)))))
        (let* ((expanded (append-map 
                         (lambda (state)
                           (map (lambda (next)
                                  (make-beam-state 
                                   (cons next (beam-state-path state))
                                   (score-fn next (beam-state-path state))))
                                (expand-fn (car (beam-state-path state)))))
                         states))
               (sorted (sort expanded (lambda (a b)
                                       (> (beam-state-score a)
                                          (beam-state-score b)))))
               (pruned (take sorted (min beam-width (length sorted)))))
          (loop pruned (1+ depth))))))

;;; Monte Carlo reasoning

(define* (monte-carlo-reasoning problem simulate-fn evaluate-fn
                                #:key (num-simulations 100))
  "Monte Carlo tree search for reasoning."
  (let* ((simulations (map (lambda (_)
                             (simulate-fn problem))
                           (iota num-simulations)))
         (evaluated (map (lambda (sim)
                          (cons sim (evaluate-fn sim)))
                        simulations))
         (best (fold (lambda (pair best)
                      (if (> (cdr pair) (cdr best))
                          pair
                          best))
                    (cons #f -inf.0)
                    evaluated)))
    (car best)))

;;; Temperature-based sampling

(define* (temperature-sampling scores #:key (temperature 1.0))
  "Sample from a distribution with temperature scaling."
  (let* ((scaled (map (lambda (s) (/ s temperature)) scores))
         (exp-scores (map exp scaled))
         (sum (apply + exp-scores))
         (probs (map (lambda (e) (/ e sum)) exp-scores))
         (rand (random:uniform)))
    (let loop ((probs probs)
               (cumsum 0)
               (index 0))
      (if (null? probs)
          (1- index)
          (let ((new-sum (+ cumsum (car probs))))
            (if (< rand new-sum)
                index
                (loop (cdr probs) new-sum (1+ index))))))))

;;; Verification methods

(define (verify-reasoning-step step context)
  "Verify a single reasoning step against context."
  (let ((description (reasoning-step-description step))
        (result (reasoning-step-result step)))
    (and (not (null? result))
         (consistent-with-context? result context))))

(define (consistent-with-context? result context)
  "Check if result is consistent with known context."
  ;; Simplified consistency check
  (not (member (cons 'contradiction result) context)))
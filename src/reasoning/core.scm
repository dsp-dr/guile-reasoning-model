;;; Core reasoning model implementation
;;; Based on "Build a Reasoning Model (From Scratch)" by Sebastian Raschka

(define-module (reasoning core)
  #:use-module (ice-9 match)
  #:use-module (ice-9 format)
  #:use-module (srfi srfi-1)
  #:use-module (srfi srfi-9)
  #:export (make-reasoning-chain
            reasoning-step
            chain-of-thought
            apply-reasoning
            reasoning-step?
            reasoning-chain?
            get-reasoning-steps
            get-final-answer))

;;; Record types for reasoning structures

(define-record-type <reasoning-step>
  (make-reasoning-step description computation result)
  reasoning-step?
  (description reasoning-step-description)
  (computation reasoning-step-computation)
  (result reasoning-step-result))

(define-record-type <reasoning-chain>
  (make-reasoning-chain* problem steps answer confidence)
  reasoning-chain?
  (problem reasoning-chain-problem)
  (steps reasoning-chain-steps)
  (answer reasoning-chain-answer)
  (confidence reasoning-chain-confidence))

;;; Chain-of-thought reasoning implementation

(define* (make-reasoning-chain problem #:key (steps '()) (answer #f) (confidence 0.0))
  "Create a new reasoning chain for a given problem."
  (make-reasoning-chain* problem steps answer confidence))

(define (chain-of-thought problem reasoning-fn)
  "Apply chain-of-thought reasoning to a problem using the given reasoning function."
  (let* ((steps (reasoning-fn problem))
         (final-answer (if (null? steps) 
                          #f 
                          (reasoning-step-result (last steps))))
         (confidence (calculate-confidence steps)))
    (make-reasoning-chain problem 
                         #:steps steps
                         #:answer final-answer
                         #:confidence confidence)))

(define (apply-reasoning chain step)
  "Add a reasoning step to an existing chain."
  (let ((current-steps (reasoning-chain-steps chain)))
    (make-reasoning-chain (reasoning-chain-problem chain)
                         #:steps (append current-steps (list step))
                         #:answer (reasoning-step-result step)
                         #:confidence (calculate-confidence 
                                      (append current-steps (list step))))))

(define (calculate-confidence steps)
  "Calculate confidence score based on reasoning steps."
  (if (null? steps)
      0.0
      (min 1.0 (* 0.2 (length steps)))))  ; Simple heuristic: more steps = higher confidence

(define (get-reasoning-steps chain)
  "Extract all reasoning steps from a chain."
  (reasoning-chain-steps chain))

(define (get-final-answer chain)
  "Get the final answer from a reasoning chain."
  (reasoning-chain-answer chain))

;;; Example reasoning patterns

(define (decompose-problem problem)
  "Decompose a complex problem into sub-problems."
  (format #t "Decomposing problem: ~a~%" problem)
  (match problem
    ((? string? s)
     (if (string-contains s "and")
         (string-split s #\and)
         (list s)))
    ((? list? l) l)
    (_ (list problem))))

(define (verify-consistency steps)
  "Check for logical consistency in reasoning steps."
  (let loop ((steps steps)
             (seen '())
             (consistent? #t))
    (if (null? steps)
        consistent?
        (let* ((current (car steps))
               (result (reasoning-step-result current)))
          (if (member result seen)
              (begin
                (format #t "Warning: Potential circular reasoning detected~%")
                #f)
              (loop (cdr steps) 
                    (cons result seen) 
                    consistent?))))))

;;; Pattern matching for reasoning types

(define (identify-reasoning-type problem)
  "Identify the type of reasoning required for a problem."
  (match problem
    ((? string? s)
     (cond
      ((string-contains-ci s "if") 'logical)
      ((string-contains-ci s "calculate") 'mathematical)
      ((string-contains-ci s "why") 'causal)
      ((string-contains-ci s "compare") 'comparative)
      (else 'general)))
    (_ 'unknown)))

;;; String utilities
(define (string-contains-ci str substr)
  "Case-insensitive substring search."
  (let ((str-lower (string-downcase str))
        (substr-lower (string-downcase substr)))
    (string-contains str-lower substr-lower)))
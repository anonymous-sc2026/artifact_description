#!/usr/bin/perl -w
@methods = ("out-of-order-discard-most-urgent", "baseline", "out-of-order-edf");
@nb_requests = (100);
@adapters=(0);
@deadline_types = ("--random-deadlines", "");

$r="0.00001,0.001,0.1";
$f="2,3,5";
foreach $i (0..20) {
    $l = $i * 50000;
    next if $l < 250000;
    foreach $a (@adapters){
        foreach $n (@nb_requests) {
            foreach $dt (@deadline_types){
                foreach $m (@methods) {
                        $cmd = "./serving_traces.py traces/BurstGPT_new_header.csv -M mistral -g 1 $dt -n $n -r $r -f $f -m $m -l $l -a $a -p -1";
                        print "$cmd\n";
                        system $cmd;
                }
            }
        }
    }
}

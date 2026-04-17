#!/usr/bin/perl -w
# this script run a a set of runs serving tarces with different profiel and diffeetn rt_scaling factors
# it stop if models gas converges (RMSE is stable)
# by script resust ar saved in monitoring_data

use BSD::Resource;

die ("Usage: $0 <num_gpus>") if ($#ARGV !=0);

$ngpu = $ARGV[0];


my $soft = 65536;
my $hard = 65536;
setrlimit(RLIMIT_NOFILE, $soft, $hard)
    or die "setrlimit failed: $!";


@methods = ("out-of-order-discard-most-urgent");
@nb_requests = (500);
@sla_factors = ("2.0");
@adapters=(0);
@deadline_types = ("");
@traces = ("../traces/azure-trace.csv",  "../traces/BurstGPT_new_header.csv");

foreach $i (0..20) {
    $l = $i * 50000;
    foreach $a (@adapters){
        foreach $f (@sla_factors) {
            foreach $n (@nb_requests) {
                foreach $dt (@deadline_types){
                    foreach $t (@traces){
                        foreach $m (@methods) {
                            $l_str = sprintf("%d:%d:1000",$l,$l+5000);
                            $cmd = "../A3_Workload_generation/serving_traces.py -M mistral --store-monitoring $t $dt -n $n -r 0.00001,0.0001,0.001,0.01,0.1,1 -f $f -m $m -l $l_str -a $a -p 100 -g $ngpu";
                            print "$cmd\n";
                            system $cmd;
                            if ($? == -1) {
                                die "failed to execute: $!";
                            }
                            $exit = $? >> 8;
                            if ($exit == 2) {
                                die "Python exited with code 2 (converged)\n";
                            }
                        }
                    }
                }
            }
        }
    }
}

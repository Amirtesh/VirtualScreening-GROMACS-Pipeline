print "Ligand_file:\t";
$ligfile = <STDIN>;
chomp $ligfile;
open(FH, $ligfile) || die "Cannot open file\n";
@arr_file = <FH>;

for ($i = 0; $i < @arr_file; $i++) {
    chomp @arr_file[$i];
    print "@arr_file[$i]\n";
    my @name = split(/\./, @arr_file[$i]);
    my $output_file = "${name[0]}_out.pdbqt";  # Create output filename
    system("smina --config conf.txt --ligand @arr_file[$i] --out $output_file");  # Run smina with output file
}

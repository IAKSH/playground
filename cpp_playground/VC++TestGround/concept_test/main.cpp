#include "print_machine.ixx"

template <rkki::ScanService Scanner, rkki::PrintService Printer>
void print_from_scanner(const Scanner& scanner, const Printer& printer) noexcept
{
	printer.print(*scanner.scan());
}

int main() noexcept
{
	rkki::PrintMachine machine;
	print_from_scanner(machine, machine);
}
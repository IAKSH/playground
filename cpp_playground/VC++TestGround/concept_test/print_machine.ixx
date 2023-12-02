#include <iostream>
#include <memory>

namespace rkki
{
	class Document
	{
	private:
		const std::string content;

	public:
		Document(std::string content) : content(content) {}
		~Document() = default;

		const std::string_view get_content() const noexcept
		{
			return content;
		}
	};

	template <typename T>
	concept PrintService = requires(T t)
	{
		{t.print(std::declval<const Document&>())} -> std::same_as<void>;
	};

	template <typename T>
	concept ScanService = requires(T t)
	{
		{t.scan()} -> std::same_as<std::unique_ptr<Document>>;
	};

	class PrintMachine
	{
	public:
		PrintMachine() = default;
		~PrintMachine() = default;

		void print(const Document& doc) noexcept
		{
			std::cout << doc.get_content() << std::endl;
		}

		std::unique_ptr<Document> scan() noexcept
		{
			return std::make_unique<Document>("Wdnmd");
		}
	};

	static_assert(PrintService<PrintMachine>&& ScanService<PrintMachine>);
}
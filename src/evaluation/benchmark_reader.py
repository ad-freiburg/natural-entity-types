class BenchmarkReader:
    @staticmethod
    def read_benchmark(filename):
        benchmark = {}
        with open(filename, "r", encoding="utf8") as file:
            for line in file:
                entity, types = line.strip("\n").split("\t")
                types = set(types.split(" "))
                benchmark[entity] = types
        return benchmark

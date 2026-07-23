// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#include "vortex/dtype.hpp"
#include <iostream>

#include <vortex.h>
#include <vortex/data_source.hpp>
#include <vortex/writer.hpp>

using namespace std::string_view_literals;
using namespace vortex;
using dtype::Nullable;

std::vector<uint8_t> bitpack(const std::vector<bool> &validity) {
    std::vector<uint8_t> out;

    return out;
}

int main() {
    const std::string_view name = "validity.vortex";

    const Session session;
    const DataType dtype = dtype::int64(Nullable);
    Writer writer = Writer::open(session, name, dtype);
    writer.push(array);
    writer.finish();

    const DataSource ds = DataSource::open(session, {name});
    Scan scan = ds.scan();
    for (Partition &partition : scan.partitions()) {
        for (Array &array : partition.batches()) {
            const Array age = array.field("age");
            const PrimitiveView<uint8_t> age_view = age.values<uint8_t>(session);
            const std::span<const uint8_t> age_values = age_view.values();
            for (uint8_t value : age_values) {
                std::cout << int(value) << " ";
            }
        }
    }
    std::cout << "\n";

    return 0;
}

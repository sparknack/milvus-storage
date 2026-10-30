// Copyright 2026 Zilliz
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <atomic>
#include <future>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include <arrow/api.h>
#include <arrow/filesystem/localfs.h>
#include <arrow/io/memory.h>
#include <arrow/util/future.h>
#include <gtest/gtest.h>
#include <parquet/arrow/writer.h>
#include <parquet/encryption/encryption.h>

#include "milvus-storage/common/arrow_util.h"
#include "milvus-storage/common/constants.h"
#include "milvus-storage/common/metadata.h"
#include "milvus-storage/format/parquet/file_reader.h"

namespace milvus_storage::test {
namespace {

struct ReadCounters {
  std::atomic<int> footer_reads{0};
  std::atomic<bool> fail_read{false};
};

class CountingBufferReader : public arrow::io::RandomAccessFile {
  public:
  CountingBufferReader(std::shared_ptr<arrow::Buffer> data, std::shared_ptr<ReadCounters> counters)
      : reader_(std::move(data)), counters_(std::move(counters)) {}

  arrow::Status Close() override { return reader_.Close(); }
  bool closed() const override { return reader_.closed(); }
  arrow::Result<int64_t> Tell() const override { return reader_.Tell(); }
  arrow::Status Seek(int64_t position) override { return reader_.Seek(position); }
  arrow::Result<int64_t> GetSize() override { return reader_.GetSize(); }
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override { return reader_.Read(nbytes, out); }
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override { return reader_.Read(nbytes); }

  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override {
    ARROW_RETURN_NOT_OK(BeforeRead(position, nbytes));
    return reader_.ReadAt(position, nbytes);
  }
  arrow::Result<int64_t> ReadAt(int64_t position, int64_t nbytes, void* out) override {
    ARROW_RETURN_NOT_OK(BeforeRead(position, nbytes));
    return reader_.ReadAt(position, nbytes, out);
  }
  arrow::Future<std::shared_ptr<arrow::Buffer>> ReadAsync(const arrow::io::IOContext&,
                                                          int64_t position,
                                                          int64_t nbytes) override {
    return arrow::Future<std::shared_ptr<arrow::Buffer>>::MakeFinished(ReadAt(position, nbytes));
  }

  private:
  arrow::Status BeforeRead(int64_t position, int64_t nbytes) {
    if (position + nbytes >= reader_.buffer()->size()) {
      ++counters_->footer_reads;
    }
    if (counters_->fail_read.exchange(false)) {
      return arrow::Status::IOError("injected read failure");
    }
    return arrow::Status::OK();
  }
  arrow::io::BufferReader reader_;
  std::shared_ptr<ReadCounters> counters_;
};

class CountingFileSystem : public arrow::fs::LocalFileSystem {
  public:
  using LocalFileSystem::OpenInputFile;
  arrow::Result<std::shared_ptr<arrow::io::RandomAccessFile>> OpenInputFile(const std::string& path) override {
    if (fail_open.exchange(false)) {
      return arrow::Status::IOError("injected open failure");
    }
    return std::make_shared<CountingBufferReader>(files.at(path), counters);
  }
  std::map<std::string, std::shared_ptr<arrow::Buffer>> files;
  std::shared_ptr<ReadCounters> counters = std::make_shared<ReadCounters>();
  std::atomic<bool> fail_open{false};
};

// Real packed Parquet bytes, with four distinguishable row groups.
arrow::Result<std::shared_ptr<arrow::Buffer>> MakePackedFile(
    int64_t base = 0, std::shared_ptr<parquet::FileEncryptionProperties> encryption = nullptr) {
  arrow::Int64Builder values;
  for (int64_t i = 0; i < 16; ++i) {
    ARROW_RETURN_NOT_OK(values.Append(base + i));
  }
  ARROW_ASSIGN_OR_RAISE(auto array, values.Finish());
  RowGroupMetadataVector row_groups;
  for (int64_t i = 0; i < 4; ++i) {
    row_groups.Add(RowGroupMetadata(32, 4, i * 4));
  }
  GroupFieldIDList groups(std::vector<FieldIDList>{FieldIDList(std::vector<FieldID>{100})});
  auto schema = arrow::schema(
      {arrow::field("value", arrow::int64(), false, arrow::key_value_metadata({ARROW_FIELD_ID_KEY}, {"100"}))},
      arrow::key_value_metadata({ROW_GROUP_META_KEY, STORAGE_VERSION_KEY, GROUP_FIELD_ID_LIST_META_KEY},
                                {row_groups.Serialize(), "1.0.0", groups.Serialize()}));
  auto table = arrow::Table::Make(schema, {array});
  ARROW_ASSIGN_OR_RAISE(auto sink, arrow::io::BufferOutputStream::Create());
  parquet::WriterProperties::Builder properties;
  if (encryption) {
    properties.encryption(std::move(encryption));
  }
  ARROW_RETURN_NOT_OK(parquet::arrow::WriteTable(*table, arrow::default_memory_pool(), sink, 4, properties.build(),
                                                 parquet::ArrowWriterProperties::Builder().store_schema()->build()));
  return sink->Finish();
}

class FileReaderMetadataTest : public ::testing::Test {
  protected:
  void SetUp() override {
    auto result = MakePackedFile();
    ASSERT_TRUE(result.ok()) << result.status();
    fs->files["file"] = *result;
  }
  arrow::Result<std::shared_ptr<FileRowGroupReader>> Open(
      std::shared_ptr<PackedFileMetadata> metadata = nullptr,
      const parquet::ReaderProperties& properties = parquet::default_reader_properties()) {
    return FileRowGroupReader::Make(fs, "file", /*buffer_size=*/32, properties,
                                    parquet::default_arrow_reader_properties(), std::move(metadata));
  }
  std::shared_ptr<CountingFileSystem> fs = std::make_shared<CountingFileSystem>();
};

TEST_F(FileReaderMetadataTest, ReusesFooterAndPackedMetadataAfterOriginalReaderCloses) {
  auto first = Open();
  ASSERT_TRUE(first.ok()) << first.status();
  auto metadata = (*first)->file_metadata();
  EXPECT_TRUE(metadata->reusable_for_reading());
  ASSERT_TRUE((*first)->Close().ok());
  const int reads = fs->counters->footer_reads;
  ASSERT_GT(reads, 0);
  auto second = Open(metadata);
  ASSERT_TRUE(second.ok()) << second.status();
  EXPECT_EQ((*second)->file_metadata(), metadata);
  EXPECT_EQ(fs->counters->footer_reads, reads);
  ASSERT_TRUE((*second)->SetRowGroupOffsetAndCount(2, 1).ok());
  std::shared_ptr<arrow::Table> table;
  ASSERT_TRUE((*second)->ReadNextRowGroup(&table).ok());
  EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0))->Value(0), 8);
  EXPECT_EQ(table->num_rows(), 4);
  auto uncached = Open();
  ASSERT_TRUE(uncached.ok()) << uncached.status();
  EXPECT_GT(fs->counters->footer_reads, reads);
}

TEST_F(FileReaderMetadataTest, ConcurrentReadersKeepIndependentCursors) {
  auto first = Open();
  ASSERT_TRUE(first.ok()) << first.status();
  auto metadata = (*first)->file_metadata();
  EXPECT_TRUE(metadata->reusable_for_reading());
  ASSERT_TRUE((*first)->Close().ok());
  const int reads = fs->counters->footer_reads;
  std::vector<std::future<void>> tasks;
  for (int i = 0; i < 16; ++i) {
    tasks.emplace_back(std::async(std::launch::async, [&, i] {
      auto result = Open(metadata);
      ASSERT_TRUE(result.ok()) << result.status();
      auto reader = *result;
      ASSERT_TRUE(reader->SetRowGroupOffsetAndCount(i % 4, 1).ok());
      std::shared_ptr<arrow::Table> table;
      ASSERT_TRUE(reader->ReadNextRowGroup(&table).ok());
      EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0))->Value(0), (i % 4) * 4);
      ASSERT_TRUE(reader->Close().ok());
    }));
  }
  for (auto& task : tasks) task.get();
  EXPECT_EQ(fs->counters->footer_reads, reads);
}

TEST_F(FileReaderMetadataTest, CachedFooterDoesNotHideDataReadFailure) {
  auto first = Open();
  ASSERT_TRUE(first.ok()) << first.status();
  auto second = Open((*first)->file_metadata());
  ASSERT_TRUE(second.ok()) << second.status();
  ASSERT_TRUE((*second)->SetRowGroupOffsetAndCount(0, 1).ok());
  fs->counters->fail_read = true;
  std::shared_ptr<arrow::Table> table;
  auto status = (*second)->ReadNextRowGroup(&table);
  EXPECT_TRUE(status.IsIOError()) << status;
  EXPECT_NE(status.message().find("injected read failure"), std::string::npos);
}

TEST_F(FileReaderMetadataTest, CachedFooterDoesNotBypassPlaintextRejection) {
  auto first = Open();
  ASSERT_TRUE(first.ok()) << first.status();
  parquet::FileDecryptionProperties::Builder decryption;
  decryption.footer_key("0123456789012345");
  auto properties = parquet::default_reader_properties();
  properties.file_decryption_properties(decryption.build());
  EXPECT_FALSE(Open(nullptr, properties).ok());
  EXPECT_FALSE(Open((*first)->file_metadata(), properties).ok());
}

TEST_F(FileReaderMetadataTest, EncryptedFilesReopenWithFreshKeys) {
  for (bool plaintext_footer : {false, true}) {
    parquet::FileEncryptionProperties::Builder encryption("0123456789012345");
    if (plaintext_footer)
      encryption.set_plaintext_footer();
    auto data = MakePackedFile(0, encryption.build());
    ASSERT_TRUE(data.ok()) << data.status();
    fs->files["file"] = *data;
    parquet::FileDecryptionProperties::Builder decryption;
    decryption.footer_key("0123456789012345");
    auto properties = parquet::default_reader_properties();
    properties.file_decryption_properties(decryption.build());
    auto first = Open(nullptr, properties);
    ASSERT_TRUE(first.ok()) << first.status();
    auto metadata = (*first)->file_metadata();
    EXPECT_FALSE(metadata->reusable_for_reading());
    ASSERT_TRUE((*first)->Close().ok());
    const int reads = fs->counters->footer_reads;
    auto second = Open(metadata, properties);
    ASSERT_TRUE(second.ok()) << second.status();
    EXPECT_NE((*second)->file_metadata(), metadata);
    EXPECT_GT(fs->counters->footer_reads, reads);
    ASSERT_TRUE((*second)->SetRowGroupOffsetAndCount(1, 1).ok());
    std::shared_ptr<arrow::Table> table;
    ASSERT_TRUE((*second)->ReadNextRowGroup(&table).ok());
    EXPECT_EQ(std::static_pointer_cast<arrow::Int64Array>(table->column(0)->chunk(0))->Value(0), 4);
    parquet::FileDecryptionProperties::Builder wrong_key;
    wrong_key.footer_key("5432109876543210");
    properties.file_decryption_properties(wrong_key.build());
    EXPECT_FALSE(Open(metadata, properties).ok());
  }
}

}  // namespace
}  // namespace milvus_storage::test

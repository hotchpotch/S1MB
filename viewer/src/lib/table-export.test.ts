import assert from 'node:assert/strict';
import test from 'node:test';
import { tableCsv, tableMarkdown } from './table-export';

test('CSV preserves displayed order, rounded scores, quotes and multiline names', () => {
  assert.equal(tableCsv({ headers: ['#', 'Model', 'Borda Score'], rows: [['2', '日本語, "Model"\nB', '50.00'], ['1', 'A', '—']] }),
    '"#","Model","Borda Score","url","huggingface"\r\n"2","日本語, ""Model""\nB","50.00","",""\r\n"1","A","—","",""\r\n');
});

test('CSV protects spreadsheet formula prefixes in model names', () => {
  const csv = tableCsv({ headers: ['Model'], rows: [['=1+1'], [' @SUM(A1)'], ['+model'], ['-model']] });
  assert.ok(csv.includes('"\'=1+1"'));
  assert.ok(csv.includes('"\' @SUM(A1)"'));
  assert.ok(csv.includes('"\'+model"'));
  assert.ok(csv.includes('"\'-model"'));
});

test('Markdown produces a plain table while escaping authored syntax', () => {
  assert.equal(tableMarkdown({ headers: ['Model', 'Score'], rows: [['A|B\\C\n<script>', '50.00'], ['[x]*_`', '—']] }),
    '| Model | Score |\n| --- | --- |\n| A\\|B\\\\C<br>&lt;script&gt; | 50.00 |\n| \\[x\\]\\*\\_\\` | — |\n');
});

test('Exports retain both URLs and prefer Hugging Face for Markdown model links', () => {
  const table = {
    headers: ['Model'], rows: [['HF'], ['Website'], ['Missing']],
    links: [
      { url: 'https://example.com', huggingface: 'https://huggingface.co/org/model' },
      { url: 'https://example.com/model(v1)', huggingface: '' },
      { url: '', huggingface: '' },
    ],
  };
  assert.equal(tableMarkdown(table), '| Model |\n| --- |\n| [HF](https://huggingface.co/org/model) |\n| [Website](https://example.com/model%28v1%29) |\n| Missing |\n');
  assert.equal(tableCsv(table), '"Model","url","huggingface"\r\n"HF","https://example.com","https://huggingface.co/org/model"\r\n"Website","https://example.com/model(v1)",""\r\n"Missing","",""\r\n');
});

#! /usr/bin/env python
# -*- coding: utf-8 -*-

""" 
This file is made by baruchel
It can be found at: https://github.com/baruchel/txt2pdf
"""

import argparse
import json
import reportlab.lib.pagesizes
from reportlab.pdfgen.canvas import Canvas
from reportlab.lib import units
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import re
import sys
import os

version_tuple = version_info = (1, 0, 2)
version = version_string = __version__ = '%d.%d.%d' % version_tuple

try:
    basestring
    py_v2 = True
    py_v3 = False
except NameError:
    basestring = str
    py_v2 = False
    py_v3 = True

def align_up(x, n):
    """Round up"""
    return ((x+n-1)//n)*n

def expand_tabs(s, tab_size=4):
    # Convert tabs to spaces, based on position in string. I.e. do not naively replace a single tab with fixed tab_size number spaces
    pos = 0
    result = []
    for c in s:
        if c == '\t':
            aligned = align_up(pos, tab_size)
            if pos % tab_size != 0:
                num_spaces = aligned - pos
            else:
                num_spaces = tab_size
            co = ' ' * num_spaces
            pos += num_spaces
        else:
            co = c
            pos += 1
        result.append(co)
    return ''.join(result)

class Margins(object):
    def __init__(self, right, left, top, bottom):
        self._right = right
        self._left = left
        self._top = top
        self._bottom = bottom

    @property
    def right(self):
        return self._right * units.cm

    @property
    def left(self):
        return self._left * units.cm

    @property
    def top(self):
        return self._top * units.cm

    @property
    def bottom(self):
        return self._bottom * units.cm

    def adjustLeft(self, width):
        self._left -= width / units.cm

class PDFCreator(object):
    appName = "txt2pdf (version %s)" % __version__

    def __init__(self, args, margins):
        pageWidth, pageHeight = reportlab.lib.pagesizes.__dict__[args.media]
        if args.landscape:
            pageWidth, pageHeight = reportlab.lib.pagesizes.landscape(
                (pageWidth, pageHeight))
        self.author = args.author
        self.title = args.title
        self.keywords = args.keywords
        self.subject = args.subject
        self.canvas = Canvas(args.output, pagesize=(pageWidth, pageHeight))
        self.canvas.setCreator(self.appName)
        if len(args.author) > 0:
            self.canvas.setAuthor(args.author)
        if len(args.title) > 0:
            self.canvas.setTitle(args.title)
        if len(args.subject) > 0:
            self.canvas.setSubject(args.subject)
        if len(args.keywords) > 0:
            self.canvas.setKeywords(args.keywords)
        self.fontSize = args.font_size
        if args.font not in ('Courier'):
            self.font = 'myFont'
            pdfmetrics.registerFont(TTFont('myFont', args.font))
        else:
            self.font = args.font
        self.kerning = args.kerning
        self.margins = margins
        self.leading = (args.extra_vertical_space + 1.2) * self.fontSize
        self.linesPerPage = int(
            (self.leading + pageHeight
             - margins.top - margins.bottom - self.fontSize) / self.leading)
        self.lppLen = len(str(self.linesPerPage))
        fontWidth = self.canvas.stringWidth(
            ".", fontName=self.font, fontSize=self.fontSize)
        self.lineNumbering = args.line_numbers
        if self.lineNumbering:
            margins.adjustLeft(fontWidth * (self.lppLen + 2))
        contentWidth = pageWidth - margins.left - margins.right
        self.charsPerLine = int(
            (contentWidth + self.kerning) / (fontWidth + self.kerning))
        self.charsWidestLineSeen = 0
        self.top = pageHeight - margins.top - self.fontSize
        self.filename = args.filename
        self.verbose = not args.quiet
        self.breakOnBlanks = args.break_on_blanks
        self.encoding = args.encoding
        self.pageNumbering = args.page_numbers
        self.tabSize = int(args.tab_size)
        self.tabReplacement = args.tab_replacement
        self.tabSeen = False
        if self.pageNumbering:
            self.pageNumberPlacement = \
               (pageWidth / 2, margins.bottom / 2)
        self.minimum_page_length = args.minimum_page_length
        if args.character_replacement:
            with open(args.character_replacement, 'rb') as f:
                json_data_str = f.read()
                self.character_replacement = json.loads(json_data_str)
                # massage data into a form string.translate() will accept,
                # i.e. convert keys that are strings into decimal
                for translate_key in self.character_replacement:
                    if isinstance(translate_key, basestring):
                        new_translate_key = ord(translate_key)  # assume it is a single character
                        self.character_replacement[new_translate_key] = self.character_replacement[translate_key]
                        del self.character_replacement[translate_key]
                print(json.dumps(self.character_replacement, indent=4))  # debug
        else:
            self.character_replacement = {}

    def _process(self, data):
        flen = os.fstat(data.fileno()).st_size
        lineno = 0
        read = 0  # number of bytes read
        for line in data:
            lineno += 1
            read += len(line)
            line = line.decode(self.encoding)
            if self.character_replacement:
                line = line.translate(self.character_replacement)  # FIXME won't work with Python 2.x when key outside of latin1 range
            if self.tabReplacement:
                line = line.replace('\t', self.tabReplacement)
            elif self.tabSize:
                line = expand_tabs(line, self.tabSize)
            elif (not self.tabSeen) and '\t' in line:
                self.tabSeen = True
            yield flen == read, lineno, line.rstrip('\r\n')

    def _readDocument(self):
        with open(self.filename, 'rb') as data:
            for done, lineno, line in self._process(data):
                lineLen = len(line)
                if lineLen > self.charsWidestLineSeen:
                    self.charsWidestLineSeen = lineLen
                if lineLen > self.charsPerLine:
                    self._scribble(
                        "Warning: wrapping line %d in %s" %
                        (lineno + 1, self.filename))
                    while len(line) > self.charsPerLine:
                        yield done, line[:self.charsPerLine]
                        line = line[self.charsPerLine:]
                yield done, line

    def _newpage(self):
        textobject = self.canvas.beginText()
        textobject.setFont(self.font, self.fontSize, leading=self.leading)
        textobject.setTextOrigin(self.margins.left, self.top)
        textobject.setCharSpace(self.kerning)
        if self.pageNumbering:
            self.canvas.drawString(
                self.pageNumberPlacement[0],
                self.pageNumberPlacement[1],
                str(self.canvas.getPageNumber()))
        return textobject

    def _scribble(self, text):
        if self.verbose:
            sys.stderr.write(text + os.linesep)

    def generate(self):
        self._scribble(
            "Writing '%s' with %d characters per "
            "line and %d lines per page..." %
            (self.filename, self.charsPerLine, self.linesPerPage)
        )
        if self.breakOnBlanks:
            pageno = self._generateBob(self._readDocument())
        else:
            pageno = self._generatePlain(self._readDocument())
        if self.charsWidestLineSeen > self.charsPerLine:
            self._scribble("Page is %d characters wide; to avoid wrapping, need at least %d" % (self.charsPerLine, self.charsWidestLineSeen))
        if self.tabSeen:
            self._scribble("Warning: Tab characters seen, but no tab-size or tab-replacement specified")
        self._scribble("PDF document: %d pages" % pageno)

    def _generatePlain(self, data):
        pageno = 1
        lineno = 0
        page = self._newpage()
        for _, line in data:
            lineno += 1

            # Handle form feed characters.
            (line, pageBreakCount) = re.subn(r'\f', r'', line)
            if pageBreakCount > 0 and lineno >= self.minimum_page_length:
                for _ in range(pageBreakCount):
                    self.canvas.drawText(page)
                    self.canvas.showPage()
                    lineno = 0
                    pageno += 1
                    page = self._newpage()
                    if self.minimum_page_length > 0:
                        break

            page.textLine(line)

            if lineno == self.linesPerPage:
                self.canvas.drawText(page)
                self.canvas.showPage()
                lineno = 0
                pageno += 1
                page = self._newpage()
        if lineno > 0:
            self.canvas.drawText(page)
        else:
            pageno -= 1
        self.canvas.save()
        return pageno

    def _writeChunk(self, page, chunk, lineno):
        if self.lineNumbering:
            formatstr = '%%%dd: %%s' % self.lppLen
            for index, line in enumerate(chunk):
                page.textLine(
                    formatstr % (lineno - len(chunk) + index + 1, line))
        else:
            for line in chunk:
                page.textLine(line)

    def _generateBob(self, data):
        pageno = 1
        lineno = 0
        page = self._newpage()
        chunk = list()
        for last, line in data:
            if lineno == self.linesPerPage:
                self.canvas.drawText(page)
                self.canvas.showPage()
                lineno = len(chunk)
                pageno += 1
                page = self._newpage()
            lineno += 1
            chunk.append(line)
            if last or len(line.strip()) == 0:
                self._writeChunk(page, chunk, lineno)
                chunk = list()
        if lineno > 0:
            self.canvas.drawText(page)
            self.canvas.showPage()
        else:
            pageno -= 1
        if len(chunk) > 0:
            page = self._newpage()
            self.canvas.drawText(page)
            self.canvas.showPage()
            pageno += 1
        self.canvas.save()
        return pageno

parser = argparse.ArgumentParser()
parser.add_argument('filename')
parser.add_argument(
    '--font',
    '-f',
    default='Courier',
    help='Select a font (True Type format) by its full path')
parser.add_argument(
    '--font-size',
    '-s',
    type=float,
    default=10.0,
    help='Size of the font')
parser.add_argument(
    '--extra-vertical-space',
    '-v',
    type=float,
    default=0.0,
    help='Extra vertical space between lines')
parser.add_argument(
    '--kerning',
    '-k',
    type=float,
    default=0.0,
    help='Extra horizontal space between characters')
parser.add_argument(
    '--media',
    '-m',
    default='A4',
    help='Select the size of the page (A4, A3, etc.)')
parser.add_argument(
    '--minimum-page-length',
    '-M',
    type=int,
    default=10,
    help='The minimum number of lines before a form feed character will change the page')
parser.add_argument(
    '--landscape',
    '-l',
    action="store_true",
    default=False,
    help='Select landscape mode')
parser.add_argument(
    '--margin-left',
    '-L',
    type=float,
    default=2.0,
    help='Left margin (in cm unit)')
parser.add_argument(
    '--margin-right',
    '-R',
    type=float,
    default=2.0,
    help='Right margin (in cm unit)')
parser.add_argument(
    '--margin-top',
    '-T',
    type=float,
    default=2.0,
    help='Top margin (in cm unit)')
parser.add_argument(
    '--margin-bottom',
    '-B',
    type=float,
    default=2.0,
    help='Bottom margin (in cm unit)')
parser.add_argument(
    '--output',
    '-o',
    default='output.pdf',
    help='Output file')
parser.add_argument(
    '--author',
    default='',
    help='Author of the PDF document')
parser.add_argument(
    '--title',
    default='',
    help='Title of the PDF document')
parser.add_argument(
    '--quiet',
    '-q',
    action='store_true',
    default=False,
    help='Hide detailed information')
parser.add_argument('--subject',default='',help='Subject of the PDF document')
parser.add_argument('--keywords',default='',help='Keywords of the PDF document')
parser.add_argument(
    '--break-on-blanks',
    '-b',
    action='store_true',
    default=False,
    help='Only break page on blank lines')
parser.add_argument(
    '--encoding',
    '-e',
    type=str,
    default='utf8',
    help='Input encoding')
parser.add_argument(
    '--page-numbers',
    '-n',
    action='store_true',
    help='Add page numbers')
parser.add_argument(
    '--line-numbers',
    action='store_true',
    help='Add line numbers')
parser.add_argument(
    '--tab-size',
    type=int,
    default=0,
    help='If not zero, replace tabs with with tab-size number of spaces')
parser.add_argument(
    '--tab-replacement',
    help='Replace tab with this character string')
parser.add_argument(
    '--character-replacement',
    '-c',
    help='Filename of json file containing mappings of replacement/translations characters')

def main(argv=None):
    if argv is None:
        argv = sys.argv

    print('Python %s on %s' % (sys.version, sys.platform))

    args = parser.parse_args()

    PDFCreator(args, Margins(
        args.margin_right,
        args.margin_left,
        args.margin_top,
        args.margin_bottom)).generate()

    return 0


if __name__ == "__main__":
    sys.exit(main())
    
def convert_txt_to_pdf(input_txt_path: str, output_pdf_path: str = "output.pdf"):
    """
    Convert a .txt file to PDF using default settings.

    Args:
        input_txt_path (str): Path to the .txt file.
        output_pdf_path (str): Desired path for the generated .pdf file.
    """
    argv = [
        input_txt_path,
        "--output", output_pdf_path,
        "--font", "Courier",
        "--font-size", "10",
        "--media", "A4",
        "--margin-left", "2",
        "--margin-right", "2",
        "--margin-top", "2",
        "--margin-bottom", "2",
        "--encoding", "utf8"
    ]
    sys.argv = [sys.argv[0]] + argv
    main(argv)
    

#![allow(clippy::comparison_chain)]

use std::collections::{HashMap, HashSet};
use std::io::{self, IsTerminal, Write as _};
use std::ops::Range;
use std::path::{Path, PathBuf, MAIN_SEPARATOR_STR};
use std::sync::{OnceLock, RwLock};
use std::fs;

use comemo::{Prehashed, Track};
use tiny_skia as sk;
use typst::diag::{bail, FileError, FileResult, Severity, SourceDiagnostic, StrResult};
use typst::foundations::{func, Bytes, Datetime, NoneValue, Repr, Smart, Value};
use typst::introspection::Meta;
use typst::layout::{Abs, Frame, FrameItem, Margin, Page, PageElem, Transform};
use typst::model::Document;
use typst::syntax::{FileId, Source, SyntaxNode, VirtualPath,PackageVersion};
use typst::text::{Font, FontBook, TextElem, TextSize};
use typst::visualize::Color;
use typst::{Library, World, WorldExt};
use walkdir::WalkDir;
use std::fmt::{self, Display, Formatter,Write};
use std::str::FromStr;

use ecow::EcoString;
// use std::env;
// use rayon::iter::{ParallelBridge, ParallelIterator};
// use oxipng::{InFile, Options, OutFile};
// use clap::Parser;
// use typst::eval::Tracer;
// use std::ffi::OsStr;
// use unscanny::Scanner;

// const TYP_DIR: &str = "typ";
// const REF_DIR: &str = "ref";
// const PNG_DIR: &str = "png";
// const PDF_DIR: &str = "pdf";
// const SVG_DIR: &str = "svg";
const FONT_DIR: &str = "../assets/fonts";
const ASSET_DIR: &str = "../assets";



/// Each test and subset may contain metadata.
#[derive(Debug)]
pub struct TestMetadata {
    /// Configures how the test is run.
    pub config: TestConfig,
    /// Declares properties that must hold for a test.
    ///
    /// For instance, `// Warning: 1-3 no text within underscores`
    /// will fail the test if the warning isn't generated by your test.
    pub annotations: HashSet<Annotation>,
}

/// Configuration of a test or subtest.
#[derive(Debug, Default)]
pub struct TestConfig {
    /// Reference images will be generated and compared.
    ///
    /// Defaults to `true`, can be disabled with `Ref: false`.
    pub compare_ref: Option<bool>,
    /// Hint annotations will be compared to compiler hints.
    ///
    /// Defaults to `true`, can be disabled with `Hints: false`.
    pub validate_hints: Option<bool>,
    /// Autocompletion annotations will be validated against autocompletions.
    /// Mutually exclusive with error and hint annotations.
    ///
    /// Defaults to `false`, can be enabled with `Autocomplete: true`.
    pub validate_autocomplete: Option<bool>,
}

/// Parsing error when the metadata is invalid.
pub(crate) enum InvalidMetadata {
    /// An invalid annotation and it's error message.
    InvalidAnnotation(Annotation, String),
    /// Setting metadata can only be done with `true` or `false` as a value.
    InvalidSet(String),
}

impl InvalidMetadata {
    pub(crate) fn write(
        invalid_data: Vec<InvalidMetadata>,
        output: &mut String,
        print_annotation: &mut impl FnMut(&Annotation, &mut String),
    ) {
        use std::fmt::Write;
        for data in invalid_data.into_iter() {
            let (annotation, error) = match data {
                InvalidMetadata::InvalidAnnotation(a, e) => (Some(a), e),
                InvalidMetadata::InvalidSet(e) => (None, e),
            };
            write!(output, "{error}",).unwrap();
            if let Some(annotation) = annotation {
                print_annotation(&annotation, output)
            } else {
                writeln!(output).unwrap();
            }
        }
    }
}

/// Annotation of the form `// KIND: RANGE TEXT`.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Annotation {
    /// Which kind of annotation this is.
    pub kind: AnnotationKind,
    /// May be written as:
    /// - `{line}:{col}-{line}:{col}`, e.g. `0:4-0:6`.
    /// - `{col}-{col}`, e.g. `4-6`:
    ///    The line is assumed to be the line after the annotation.
    /// - `-1`: Produces a range of length zero at the end of the next line.
    ///   Mostly useful for autocompletion tests which require an index.
    pub range: Option<Range<usize>>,
    /// The raw text after the annotation.
    pub text: EcoString,
}

/// The different kinds of in-test annotations.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum AnnotationKind {
    Error,
    Warning,
    Hint,
    AutocompleteContains,
    AutocompleteExcludes,
}

impl AnnotationKind {
    /// Returns the user-facing string for this annotation.
    pub fn as_str(self) -> &'static str {
        match self {
            AnnotationKind::Error => "Error",
            AnnotationKind::Warning => "Warning",
            AnnotationKind::Hint => "Hint",
            AnnotationKind::AutocompleteContains => "Autocomplete contains",
            AnnotationKind::AutocompleteExcludes => "Autocomplete excludes",
        }
    }
}

impl FromStr for AnnotationKind {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(match s {
            "Error" => AnnotationKind::Error,
            "Warning" => AnnotationKind::Warning,
            "Hint" => AnnotationKind::Hint,
            "Autocomplete contains" => AnnotationKind::AutocompleteContains,
            "Autocomplete excludes" => AnnotationKind::AutocompleteExcludes,
            _ => return Err("invalid annotatino"),
        })
    }
}

impl Display for AnnotationKind {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.pad(self.as_str())
    }
}


fn library() -> Library {
    #[func]
    fn test(lhs: Value, rhs: Value) -> StrResult<NoneValue> {
        if lhs != rhs {
            bail!("Assertion failed: {} != {}", lhs.repr(), rhs.repr());
        }
        Ok(NoneValue)
    }

    #[func]
    fn test_repr(lhs: Value, rhs: Value) -> StrResult<NoneValue> {
        if lhs.repr() != rhs.repr() {
            bail!("Assertion failed: {} != {}", lhs.repr(), rhs.repr());
        }
        Ok(NoneValue)
    }

    #[func]
    fn print(#[variadic] values: Vec<Value>) -> NoneValue {
        let mut stdout = io::stdout().lock();
        write!(stdout, "> ").unwrap();
        for (i, value) in values.into_iter().enumerate() {
            if i > 0 {
                write!(stdout, ", ").unwrap();
            }
            write!(stdout, "{value:?}").unwrap();
        }
        writeln!(stdout).unwrap();
        NoneValue
    }

    // Set page width to 120pt with 10pt margins, so that the inner page is
    // exactly 100pt wide. Page height is unbounded and font size is 10pt so
    // that it multiplies to nice round numbers.
    let mut lib = Library::default();
    lib.styles
        .set(PageElem::set_width(Smart::Custom(Abs::pt(120.0).into())));
    lib.styles.set(PageElem::set_height(Smart::Auto));
    lib.styles.set(PageElem::set_margin(Margin::splat(Some(Smart::Custom(
        Abs::pt(10.0).into(),
    )))));
    lib.styles.set(TextElem::set_size(TextSize(Abs::pt(10.0).into())));

    // Hook up helpers into the global scope.
    lib.global.scope_mut().define_func::<test>();
    lib.global.scope_mut().define_func::<test_repr>();
    lib.global.scope_mut().define_func::<print>();
    lib.global
        .scope_mut()
        .define("conifer", Color::from_u8(0x9f, 0xEB, 0x52, 0xFF));
    lib.global
        .scope_mut()
        .define("forest", Color::from_u8(0x43, 0xA1, 0x27, 0xFF));

    lib
}

fn system_path(id: FileId) -> FileResult<PathBuf> {
    let root: PathBuf = match id.package() {
        Some(spec) => {println!("{} {}",spec.name, spec.version); format!("packages/{}-{}", spec.name, spec.version).into()},
        None => PathBuf::new(),
    };

    id.vpath().resolve(&root).ok_or(FileError::AccessDenied)
}

/// Draw all frames into one image with padding in between.
pub fn render(document: &Document) -> sk::Pixmap {
    let pixel_per_pt = 2.0;
    let padding = Abs::pt(5.0);

    for page in &document.pages {
        let limit = Abs::cm(100.0);
        if page.frame.width() > limit || page.frame.height() > limit {
            panic!("overlarge frame: {:?}", page.frame.size());
        }
    }

    let mut pixmap = typst_render::render_merged(
        document,
        pixel_per_pt,
        Color::WHITE,
        padding,
        Color::BLACK,
    );

    let padding = (pixel_per_pt * padding.to_pt() as f32).round();
    let [x, mut y] = [padding; 2];
    for page in &document.pages {
        let ts =
            sk::Transform::from_scale(pixel_per_pt, pixel_per_pt).post_translate(x, y);
        render_links(&mut pixmap, ts, &page.frame);
        y += (pixel_per_pt * page.frame.height().to_pt() as f32).round().max(1.0)
            + padding;
    }

    pixmap
}

/// Draw extra boxes for links so we can see whether they are there.
fn render_links(canvas: &mut sk::Pixmap, ts: sk::Transform, frame: &Frame) {
    for (pos, item) in frame.items() {
        let ts = ts.pre_translate(pos.x.to_pt() as f32, pos.y.to_pt() as f32);
        match *item {
            FrameItem::Group(ref group) => {
                let ts = ts.pre_concat(to_sk_transform(&group.transform));
                render_links(canvas, ts, &group.frame);
            }
            FrameItem::Meta(Meta::Link(_), size) => {
                let w = size.x.to_pt() as f32;
                let h = size.y.to_pt() as f32;
                let rect = sk::Rect::from_xywh(0.0, 0.0, w, h).unwrap();
                let mut paint = sk::Paint::default();
                paint.set_color_rgba8(40, 54, 99, 40);
                canvas.fill_rect(rect, &paint, ts, None);
            }
            _ => {}
        }
    }
}

fn to_sk_transform(transform: &Transform) -> sk::Transform {
    let Transform { sx, ky, kx, sy, tx, ty } = *transform;
    sk::Transform::from_row(
        sx.get() as _,
        ky.get() as _,
        kx.get() as _,
        sy.get() as _,
        tx.to_pt() as f32,
        ty.to_pt() as f32,
    )
}

pub struct TestWorld {
    main: FileId,
    library: Prehashed<Library>,
    book: Prehashed<FontBook>,
    fonts: Vec<Font>,
    slots: RwLock<HashMap<FileId, FileSlot>>,
}

#[derive(Clone)]
struct FileSlot {
    source: OnceLock<FileResult<Source>>,
    buffer: OnceLock<FileResult<Bytes>>,
}

impl TestWorld {
    pub fn new() -> Self {
        // Search for fonts.
        let mut fonts = vec![];
        for entry in WalkDir::new(FONT_DIR)
            .sort_by_file_name()
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|entry| entry.file_type().is_file())
        {
            let data = fs::read(entry.path()).unwrap();
            fonts.extend(Font::iter(data.into()));
        }

        Self {
            main: FileId::new(None, VirtualPath::new("main.typ")),
            library: Prehashed::new(library()),
            book: Prehashed::new(FontBook::from_fonts(&fonts)),
            fonts,
            slots: RwLock::new(HashMap::new()),
        }
    }
}

impl World for TestWorld {
    fn library(&self) -> &Prehashed<Library> {
        &self.library
    }

    fn book(&self) -> &Prehashed<FontBook> {
        &self.book
    }

    fn main(&self) -> Source {
        self.source(self.main).unwrap()
    }

    fn source(&self, id: FileId) -> FileResult<Source> {
        self.slot(id, |slot| {
            slot.source
                .get_or_init(|| {
                    let buf = read(&system_path(id)?)?;
                    let text = String::from_utf8(buf)?;
                    Ok(Source::new(id, text))
                })
                .clone()
        })
    }

    fn file(&self, id: FileId) -> FileResult<Bytes> {
        self.slot(id, |slot| {
            slot.buffer
                .get_or_init(|| read(&system_path(id)?).map(Bytes::from))
                .clone()
        })
    }

    fn font(&self, id: usize) -> Option<Font> {
        Some(self.fonts[id].clone())
    }

    fn today(&self, _: Option<i64>) -> Option<Datetime> {
        Some(Datetime::from_ymd(1970, 1, 1).unwrap())
    }
}

impl TestWorld {
    pub fn set(&mut self, path: &Path, text: String) -> Source {
        self.main = FileId::new(None, VirtualPath::new(path));
        let source = Source::new(self.main, text);
        self.slot(self.main, |slot| {
            slot.source = OnceLock::from(Ok(source.clone()));
            source
        })
    }

    fn slot<F, T>(&self, id: FileId, f: F) -> T
    where
        F: FnOnce(&mut FileSlot) -> T,
    {
        f(self.slots.write().unwrap().entry(id).or_insert_with(|| FileSlot {
            source: OnceLock::new(),
            buffer: OnceLock::new(),
        }))
    }
}

impl Clone for TestWorld {
    fn clone(&self) -> Self {
        Self {
            main: self.main,
            library: self.library.clone(),
            book: self.book.clone(),
            fonts: self.fonts.clone(),
            slots: RwLock::new(self.slots.read().unwrap().clone()),
        }
    }
}

fn read(path: &Path) -> FileResult<Vec<u8>> {
    // Basically symlinks `assets/files` to `tests/files` so that the assets
    // are within the test project root.
    let mut resolved = path.to_path_buf();
    if path.starts_with("files/") {
        resolved = Path::new(ASSET_DIR).join(path);
    }

    let f = |e| FileError::from_io(e, path);
    if fs::metadata(&resolved).map_err(f)?.is_dir() {
        Err(FileError::IsDirectory)
    } else {
        fs::read(&resolved).map_err(f)
    }
}
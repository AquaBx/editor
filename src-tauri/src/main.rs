// Prevents additional console window on Windows in release, DO NOT REMOVE!!
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use serde::Serialize;
use std::fs;
use std::path::Path;

mod myworld;

#[derive(Serialize, Clone, Debug)]
pub struct TypstRenderResponse {
    pub buffer: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

#[tauri::command]
fn my_typst_render() -> TypstRenderResponse  { 

    let mut world = crate::myworld::TestWorld::new();

    let out = Path::new("E:/img.png");
    let path = Path::new("E:/GitRepo/typst-editor/src-tauri/src/files");
    let srcpath = Path::new("E:/GitRepo/typst-editor/src-tauri/src/files/main.typ");
    let text = fs::read_to_string(srcpath).unwrap();

    world.set(&path, text);

    let mut pages = vec![];
    let mut tracer = typst::eval::Tracer::new();

    match typst::compile(&world, &mut tracer) {
        Ok(document) => pages.extend(document.pages),
        Err(e) => println!("{:#?}",e),
    };
    let document =  typst::model::Document { pages, ..Default::default() };

    let bmp = crate::myworld::render(&document);

    let e = bmp.save_png(out);
    
    println!("{:#?}",e);

    match bmp.encode_png() {
        Ok(image) => return TypstRenderResponse { buffer : image, width : bmp.width(), height : bmp.height() },
        Err(_) => return TypstRenderResponse{width:0,height:0,buffer:vec!{}}
    }
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![my_typst_render])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}


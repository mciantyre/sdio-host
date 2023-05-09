//! Blocking SDIO host abstraction.

use core::cell::RefCell;

use crate::{
    common_cmd::{
        self, all_send_cid, card_status, idle, read_single_block, select_card, send_csd,
        set_block_length, write_single_block, Resp,
    },
    sd::{BusWidth, CardStatus, CurrentState, SDStatus, CID, CSD, OCR, RCA, SCR, SD},
    sd_cmd::{self, send_if_cond, send_relative_address, send_scr, set_bus_width},
    Cmd,
};

/// Errors from the transport layer.
///
/// Although this library only defines a blocking transport, implementations
/// are free to implement that behavior using non-blocking operations, including
/// SDMA, ADMA1, and / or ADMA2. These kinds of transports may have buffer sizing
/// and alignment requirements, and these errors should find their way to the user.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum TransportError {
    /// The expected response for a command was not received.
    ///
    /// This does not apply for commands that do not expect responses.
    CommandTimeout,
    /// The expected data was not received.
    ///
    /// This does not apply for commands that do not expect data.
    DataTimeout,
    /// A CRC did not match its expected value.
    Crc,
    /// The command index in the response did not match the sent command.
    CommandIndex,
    /// A start, transmission, or end bit was not correct.
    Bit,
    /// The transport is busy.
    ///
    /// This could be due to a previous transaction that has not yet completed,
    /// or due to a signal of "busy" on a data line.
    Busy,
    /// The data buffer is not aligned.
    ///
    /// The read / write buffer supplied for the transfer is not properly
    /// aligned. Check your driver's documentation for more information.
    MisalignedBuffer {
        /// The alignment required for the data transfer.
        required_alignment: u32,
    },
    /// The data buffer is not properly sized.
    MissizedBuffer {
        /// The minimum buffer size required for the transfer.
        minimum_size: u32,
    },
    /// Operation not supported.
    NotSupported,
    /// Some other, driver-specific error.
    ///
    /// Optionally include a human-readable error message.
    Other(Option<&'static str>),
}

impl TransportError {
    /// Returns a driver-specific error with a human-readable message.
    pub const fn driver(what: &'static str) -> TransportError {
        TransportError::Other(Some(what))
    }
    /// Returns another, uncategorized error.
    ///
    /// Prefer [`driver()`] if you could give a better indication
    /// of what happened.
    pub const fn uncategorized() -> TransportError {
        TransportError::Other(None)
    }
}

/// Transport operating modes.
///
/// An implementation may select a lower clock speed than the maximum
/// clock speed.
#[derive(Debug)]
#[non_exhaustive]
pub enum TransportMode {
    /// Execute in identification mode
    ///
    /// The maximum clock speed is 400KHz.
    Identification,
    /// Execute in SDIO full-speed mode.
    ///
    /// The maximum clock speed is 25MHz.
    SdFullSpeed,
}

/// Data to send to / receive from a card.
///
/// The caller may associate data to be transported with
/// a command. A `None` indicates that there is no additional
/// data.
///
/// An implementation may have requirements for the buffer
/// size and alignment. These requirements may be coupled to
/// a command.
#[derive(Debug)]
pub enum TransportData<'b> {
    /// This transfer does not include data.
    None,
    /// Read data from the card.
    ///
    /// This is a card-to-host transfer.
    Read {
        /// Storage for the result, with length indicating
        /// the expected size.
        buffer: &'b mut [u8],
    },
    /// Write data from the card.
    ///
    /// This is a host-to-card transfer.
    Write {
        /// The data, with length indicating the expected
        /// size.
        buffer: &'b [u8],
    },
}

impl TransportData<'_> {
    /// `true` if there is no data.
    pub const fn is_none(&self) -> bool {
        matches!(self, TransportData::None)
    }
    /// `true` if the transport should read data.
    pub const fn is_read(&self) -> bool {
        matches!(self, TransportData::Read { .. })
    }
    /// `true` if the transport should write data.
    pub const fn is_write(&self) -> bool {
        matches!(self, TransportData::Write { .. })
    }
    /// Returns the number of *bytes*, if available.
    ///
    /// Returns `0` for `None`. Otherwise, returns the
    /// length of the slice.
    pub fn len(&self) -> usize {
        match self {
            Self::None => 0,
            Self::Read { buffer, .. } => buffer.len(),
            Self::Write { buffer, .. } => buffer.len(),
        }
    }
}

/// A blocking SDIO host transport.
///
/// # General implementation notes
///
/// Implementations ensure that command and data CRCs match their expected values.
/// If a CRC does not match, the implementation returns [`TransportError::Crc`].
/// Implementations may perform this check in hardware or software.
/// TODO provide CRC7, CRC16 algorithms in this package for implementations that
/// do not do this in hardware.
pub trait BlockingSdioTransport {
    /// Use the transport to send commands, receive responses, and transfer data.
    ///
    /// `transfer()` blocks until the device provides a response. Implementations
    /// may return [`TransportError::CommandTimeout`] if the device never responds.
    /// Implementations are free to define their own timeout and means of realizing
    /// the timeout.
    ///
    /// # Protocol checks
    ///
    /// The command describes the characteristics of the response, including the length,
    /// if a command index is available, and if a CRC is available. Given this information,
    /// `transfer()` ensures that general protocol details are correct. The rest of
    /// this section describes how an implementation evaluates each response.
    ///
    /// No matter the response type, all `transfer()` implementations check the start,
    /// transmission, and end bits. If the bits do not match their expected value, the
    /// implementation returns [`TransportError::Bit`].
    ///
    /// If the response includes the command index (R1, R1b, R6, and R7), then `transfer()`
    /// ensures that the response includes the correct command index. If the indices do
    /// not match, the implementation returns [`TransportError::CommandIndex`].
    ///
    /// If the response includes a CRC7, then `transfer()` ensures that the CRCs match.
    /// See the general implementation notes for more information.
    ///
    /// # Response serialization
    ///
    /// The caller always provides four words to hold response data, and `transfer()`
    /// is responsible for placing the response data into these buffer elements. This
    /// section describes how an implementation serializes the response data for the
    /// caller.
    ///
    /// | Response types       | Response Bits | `response` location |
    /// | -------------------- | ------------- | ------------------- |
    /// | R1(b), R3, R5(b), R6 |    `[39:8]`   |    `response[0]`    |
    /// | R2                   |    `[31:0]`   |    `response[0]`    |
    /// |                      |    `[63:32]`  |    `response[1]`    |
    /// |                      |    `[95:64]`  |    `response[2]`    |
    /// |                      |    `[127:96]` |    `response[3]`    |
    /// | None                 |    N/A        |        N/A          |
    ///
    /// For R1(b), R3, R5, and R6 responses, the implementation extracts the meaningful
    /// response information from bits `[39:8]` and places it into `response[0]`. The
    /// implementation understands this condition by checking if
    /// [`command.response_len()`](crate::common_cmd::Cmd::response_len)
    /// equals [`ResponseLen::R48`](crate::common_cmd::ResponseLen).
    ///
    /// For R2 responses, the implementation includes the (internal) CRC and the end bit
    /// in the response. If hardware does not expose this information, an implementation
    /// can provide unspecified, meaningless values for these fields.
    ///
    /// # Data transfer
    ///
    /// When `data` is [`TransportData::Read`] ([`TransportData::Write`]), the transport
    /// reads (writes) the associated slice of bytes. When handling reads, the transport
    /// blocks until all data is received. When handling writes, the transport may block,
    /// or it may return once the data is scheduled for transmission (by enqueueing the
    /// data into an internal buffer, for instance). In the case of an early return, the
    /// transport must block if `transfer` is called a second time the data from the
    /// previous write has not been transmitted to the device.
    ///
    /// The transport may choose any approach to schedule the data transfer. If the data
    /// is not aligned, or if the buffer is not sized properly, the transport returns
    /// [`TransportError::MisalignedBuffer`] or [`TransportError::MissizedBuffer`],
    /// respectively.
    fn transfer<R>(
        &mut self,
        command: &Cmd<R>,
        response: &mut [u32; 4],
        data: TransportData<'_>,
    ) -> Result<(), TransportError>
    where
        R: Resp;

    /// Set the bus width for data transfers.
    ///
    /// Implementations are always expected to support 1-bit transport. If an implementation
    /// does not support larger widths, it returns [`TransportError::NotSupported`].
    ///
    /// The transport should not negotiate this value with a device; that has already happened
    /// by a caller.
    fn set_bus_width(&mut self, bus_width: BusWidth) -> Result<(), TransportError>;

    /// Set the transport mode.
    ///
    /// See [`TransportMode`] for more information. The transport should not negotiate this
    /// value with a device; that has already happened by a caller.
    fn set_mode(&mut self, mode: TransportMode) -> Result<(), TransportError>;

    /// Perform a power cycle of the card.
    ///
    /// This includes:
    ///
    /// 1. Toggling the reset line.
    /// 2. Waiting for stable supply voltage.
    /// 3. Sending at least 74 clock cycles.
    ///
    /// Implementations are free to perform a software or peripheral reset during this call.
    fn power_cycle(&mut self, dela_ms: &mut impl FnMut(u32)) -> Result<(), TransportError>;
}

/// Errors during host operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum HostError {
    /// This card is not supported.
    ///
    /// This could indicate a compatibility issue with your card and this
    /// driver.
    UnsupportedCard,
    /// The card should expect an app command, but it's not.
    ///
    /// After sending CMD55, the card's status should indicate that
    /// it is expecting an application command. If that status is
    /// not correct, then the initialization process returns this
    /// error.
    ShouldExpectAppCmd,
    /// The timeout for initialization was exceeded.
    ///
    /// The specification requires that cards complete initialization
    /// within one second. If you see this error, then the card didn't
    /// respond in time.
    InitializationTimeout,
    /// An error propagated from the transport.
    Transport {
        /// If the error occured due to a command transfer,
        /// this holds the associated command index.
        command_index: Option<u8>,
        /// The transport error.
        error: TransportError,
    },
}

/// For error mapping a command tranfser to the most-helpful host error.
fn cmd_err<'a, R: Resp>(cmd: &'a Cmd<R>) -> impl FnOnce(TransportError) -> HostError + 'a {
    move |error| HostError::Transport {
        command_index: Some(cmd.cmd),
        error,
    }
}

/// For error mapping when there is no associated command.
fn transport_err() -> impl FnOnce(TransportError) -> HostError {
    |error| HostError::Transport {
        command_index: None,
        error,
    }
}

fn app_cmd<T>(transport: &mut T, rca: u16) -> Result<(), HostError>
where
    T: BlockingSdioTransport,
{
    let cmd = &common_cmd::app_cmd(rca);
    let mut response = [0; 4];
    transport
        .transfer(cmd, &mut response, TransportData::None)
        .map_err(cmd_err(cmd))?;

    let card_status = CardStatus::<SD>::from(response[0]);
    // TODO handle other card status errors...
    if !card_status.app_cmd() {
        Err(HostError::ShouldExpectAppCmd)
    } else {
        Ok(())
    }
}

fn acmd41<T>(transport: &mut T, rca: u16, delay: &mut impl FnMut(u32)) -> Result<(), HostError>
where
    T: BlockingSdioTransport,
{
    let mut response = [0; 4];

    // Note that this loop attempts, plus the delay, is around 1 second.
    let mut attempts = 100u32;
    while 0 != attempts {
        app_cmd(transport, rca)?;

        // TODO hard-coded flags, voltages should be derived from runtime / transport
        // values.
        let cmd = &sd_cmd::sd_send_op_cond(
            true, // TODO only because we assume CMD8 response.
            false, false, 0x1FF,
        );
        transport
            .transfer(cmd, &mut response, TransportData::None)
            .map_err(cmd_err(cmd))?;

        let ocr = OCR::<SD>::from(response[0]);

        if !ocr.is_busy() {
            break;
        }

        attempts -= 1;
        delay(10);
    }

    if 0 == attempts {
        Err(HostError::InitializationTimeout)
    } else {
        Ok(())
    }
}

fn get_scr<T>(transport: &mut T, rca: u16) -> Result<SCR, HostError>
where
    T: BlockingSdioTransport,
{
    let mut response = [0u32; 4];

    // TODO check response...?
    let cmd = &set_block_length(8);
    transport
        .transfer(cmd, &mut response, TransportData::None)
        .map_err(cmd_err(cmd))?;

    app_cmd(transport, rca)?;

    let cmd = &send_scr();
    let mut buffer = [0u8; 8];
    transport
        .transfer(
            cmd,
            &mut response,
            TransportData::Read {
                buffer: &mut buffer,
            },
        )
        .map_err(cmd_err(cmd))?;

    Ok(SCR(u64::from_be_bytes(buffer)))
}

fn read_sd_status<T>(transport: &mut T, rca: u16) -> Result<SDStatus, HostError>
where
    T: BlockingSdioTransport,
{
    let cmd = &set_block_length(64);
    let mut response = [0u32; 4];
    // TODO check response...?
    transport
        .transfer(cmd, &mut response, TransportData::None)
        .map_err(cmd_err(cmd))?;

    app_cmd(transport, rca)?;

    let cmd = &card_status(rca, false);
    let mut buffer = [0u8; 64];
    transport
        .transfer(
            cmd,
            &mut response,
            TransportData::Read {
                buffer: &mut buffer,
            },
        )
        .map_err(cmd_err(cmd))?;

    let mut status = [0u32; 16];
    for (bytes, word) in buffer.chunks_exact(4).zip(status.iter_mut().rev()) {
        use core::convert::TryInto;
        *word = u32::from_be_bytes(bytes.try_into().unwrap());
    }

    Ok(SDStatus::from(status))
}

fn read_status<T>(transport: &mut T, rca: u16) -> Result<CardStatus<SD>, HostError>
where
    T: BlockingSdioTransport,
{
    let response = &mut [0; 4];

    let cmd = &card_status(rca, false);
    transport
        .transfer(&cmd, response, TransportData::None)
        .map_err(cmd_err(cmd))?;

    Ok(CardStatus::from(response[0]))
}

/// An SDIO host that performs blocking operations.
pub struct BlockingSdioHost<T> {
    transport: T,
    // TODO shouldn't hard-coded SD assumptions...
    cid: CID<SD>,
    rca: RCA<SD>,
    csd: CSD<SD>,
    scr: SCR,
    sd_status: SDStatus,
}

impl<T> BlockingSdioHost<T>
where
    T: BlockingSdioTransport,
{
    /// Create a new SDIO host.
    pub fn new(mut transport: T, delay: &mut impl FnMut(u32)) -> Result<Self, HostError> {
        transport.power_cycle(delay).map_err(transport_err())?;
        transport
            .set_bus_width(BusWidth::One)
            .map_err(transport_err())?;
        transport
            .set_mode(TransportMode::Identification)
            .map_err(transport_err())?;

        let mut response = [0; 4];

        let cmd = &idle();
        transport
            .transfer(cmd, &mut response, TransportData::None)
            .map_err(cmd_err(cmd))?;

        // TODO handle the fact that not all cards respond to this...
        let cmd = &send_if_cond(0x1, 0xAA);
        transport
            .transfer(cmd, &mut response, TransportData::None)
            .map_err(cmd_err(cmd))?;
        if response[0] & 0xFF != 0xAA {
            return Err(HostError::UnsupportedCard);
        }

        acmd41(&mut transport, 0, delay)?;

        let cmd = &all_send_cid();
        transport
            .transfer(cmd, &mut response, TransportData::None)
            .map_err(cmd_err(cmd))?;
        let cid = CID::from(response);

        let cmd = &send_relative_address();
        transport
            .transfer(cmd, &mut response, TransportData::None)
            .map_err(cmd_err(cmd))?;
        let rca = RCA::from(response[0]);

        let cmd = &send_csd(rca.address());
        transport
            .transfer(cmd, &mut response, TransportData::None)
            .map_err(cmd_err(cmd))?;
        let csd = CSD::from(response);

        // TODO should check R1 response bits...
        let cmd = &select_card(rca.address());
        transport
            .transfer(cmd, &mut response, TransportData::None)
            .map_err(cmd_err(cmd))?;

        let scr = get_scr(&mut transport, rca.address())?;

        if scr.bus_width_four() && transport.set_bus_width(BusWidth::Four).is_ok() {
            app_cmd(&mut transport, rca.address())?;
            let cmd = &set_bus_width(true);
            transport
                .transfer(cmd, &mut response, TransportData::None)
                .map_err(cmd_err(cmd))?;
        }

        transport
            .set_mode(TransportMode::SdFullSpeed)
            .map_err(transport_err())?;

        let sd_status = read_sd_status(&mut transport, rca.address())?;

        Ok(Self {
            transport,
            cid,
            rca,
            csd,
            scr,
            sd_status,
        })
    }

    /// Return the card identification register contents.
    pub fn cid(&self) -> CID<SD> {
        self.cid
    }

    /// Returns the relative card address.
    pub fn rca(&self) -> RCA<SD> {
        self.rca
    }

    /// Returns card-specific data.
    pub fn csd(&self) -> CSD<SD> {
        self.csd
    }

    /// Returns the SD card configuration register.
    pub fn scr(&self) -> SCR {
        self.scr
    }

    /// Returns the SD status.
    pub fn sd_status(&self) -> &SDStatus {
        &self.sd_status
    }

    /// Release the transport from the host.
    ///
    /// The transport state is unspecified.
    pub fn release(self) -> T {
        self.transport
    }

    /// Poll the card status for card ready.
    fn card_ready(&mut self) -> Result<bool, HostError> {
        let status = self.read_status()?;
        Ok(status.state() == CurrentState::Transfer)
    }

    fn read_status(&mut self) -> Result<CardStatus<SD>, HostError> {
        read_status(&mut self.transport, self.rca.address())
    }

    /// Read a single block of data.
    ///
    /// `address` is the block index.
    pub fn read_block(&mut self, address: u32, buffer: &mut [u8; 512]) -> Result<(), HostError> {
        let response = &mut [0; 4];

        let cmd = &set_block_length(buffer.len() as u32);
        self.transport
            .transfer(&cmd, response, TransportData::None)
            .map_err(cmd_err(cmd))?;

        let cmd = &read_single_block(address);
        self.transport
            .transfer(
                &cmd,
                response,
                TransportData::Read {
                    buffer: buffer.as_mut_slice(),
                },
            )
            .map_err(cmd_err(cmd))?;

        Ok(())
    }

    /// Write a single block of data.
    ///
    /// `address` is the block index.
    pub fn write_block(&mut self, address: u32, buffer: &[u8; 512]) -> Result<(), HostError> {
        let response = &mut [0; 4];

        let cmd = &set_block_length(buffer.len() as u32);
        self.transport
            .transfer(&cmd, response, TransportData::None)
            .map_err(cmd_err(cmd))?;

        let cmd = &write_single_block(address);
        self.transport
            .transfer(
                &cmd,
                response,
                TransportData::Write {
                    buffer: buffer.as_slice(),
                },
            )
            .map_err(cmd_err(cmd))?;

        // TODO timeout...
        while !self.card_ready()? {}

        Ok(())
    }

    /// Convert the blocking host into an `embedded-sdmmc` block device.
    pub fn into_sdmmc_block_dev(self) -> SdmmcBlockDevice<T> {
        SdmmcBlockDevice(RefCell::new(self))
    }
}

pub struct SdmmcBlockDevice<T>(RefCell<BlockingSdioHost<T>>);

impl<T> embedded_sdmmc::blockdevice::BlockDevice for SdmmcBlockDevice<T>
where
    T: BlockingSdioTransport,
{
    type Error = HostError;

    fn read(
        &self,
        blocks: &mut [embedded_sdmmc::Block],
        start_block_idx: embedded_sdmmc::BlockIdx,
        _: &str,
    ) -> Result<(), Self::Error> {
        let start = start_block_idx.0;
        let mut host = self.0.borrow_mut();
        for idx in start..(start + blocks.len() as u32) {
            host.read_block(idx, &mut blocks[(idx - start) as usize].contents)?;
        }
        Ok(())
    }

    fn write(
        &self,
        blocks: &[embedded_sdmmc::Block],
        start_block_idx: embedded_sdmmc::BlockIdx,
    ) -> Result<(), Self::Error> {
        let start = start_block_idx.0;
        let mut host = self.0.borrow_mut();
        for idx in start..(start + blocks.len() as u32) {
            host.write_block(idx, &blocks[(idx - start) as usize].contents)?;
        }
        Ok(())
    }

    fn num_blocks(&self) -> Result<embedded_sdmmc::BlockCount, Self::Error> {
        let host = self.0.borrow();
        Ok(embedded_sdmmc::BlockCount(host.csd.block_count() as u32))
    }
}
